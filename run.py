import argparse
import time
import os

from comet_ml import Experiment

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from input import BabiTaskDataset
from model import EntityNetwork
from clr_scheduler import CyclicalLRScheduler
from metrics import accuracy, f1
from utils import save_model, load_model, write_summaries, prune_dump_files, prune_summaries_files

parser = argparse.ArgumentParser()

# Runtime parameters
parser.add_argument(
    "--dataset_folder_path", default="/Users/rodolfo/U/Magister/datasets/tasks_1-20_v1-2/en_tokenized",
    help="Path to the folder containing the preprocessed dataset.")
parser.add_argument(
    "--task_id", default=1, type=int,
    help="Task identifier of the bAbI task to run.")
parser.add_argument(
    "--jointly_preprocessed", default=False, type=bool,
    help="Whether the task was jointly preprocessed or not.")
parser.add_argument(
    "--epochs", default=200, type=int,
    help="Number of epochs to train.")
parser.add_argument(
    "--batch_size", default=32, type=int,
    help="Size of the training batches.")
parser.add_argument(
    "--lr_scheduler", default="step", choices=("step", "cyclical"),
    help="Learning rate scheduler.")
# For the step scheduler...
parser.add_argument(
    "--learning_rate", default=0.01, type=float,
    help="Learning rate of the training.")
parser.add_argument(
    "--decay_rate", default=0.5,
    help="How much to decay the learning rate between decay epochs.")
parser.add_argument(
    "--decay_period", default=25, type=int,
    help="Epochs between learning rate decayment.")
# For the cyclical scheduler...
parser.add_argument(
    "--min_lr", default=1e-4, type=float,
    help="Minimum learning rate for the cyclical scheduling.")
parser.add_argument(
    "--max_lr", default=5*1e-3, type=float,
    help="Maximum learning rate for the cyclical scheduling.")
parser.add_argument(
    "--cycle_period", default=5, type=int,
    help="Length of a cycle in epochs.")
# /cyclical scheduler
parser.add_argument(
    "--gradient_clipping", default=40, type=float,
    help="Upper bound for the gradients.")
parser.add_argument(
    "--validation_ratio", default=0.1, type=float,
    help="Validation set ratio.")
parser.add_argument(
    "--checkpoints_folder_path", default="/tmp/x-module/", type=str,
    help="Path to the folder where the checkpoints are going to be stored.")
parser.add_argument(
    "--runs", default=1, type=int,
    help="Number of times that the training is repeated.")
parser.add_argument(
    "--dropout_prob", default=0, type=float,
    help="Probability of dropping out elements in the input module while training.")
parser.add_argument(
    "--qa_lambda", default=1, type=float,
    help="Weight of the question answering loss.")
parser.add_argument(
    "--supporting_facts_lambda", default=1, type=float,
    help="Weight of the supporting facts loss.")
parser.add_argument(
    "--weight_decay", default=0, type=float,
    help="Adam optimizer weight decay.")
parser.add_argument(
    "--f1_threshold", default=0.5, type=float,
    help="F1 score threshold")
parser.add_argument(
    "--teach_force_training", default=False, action="store_true",
    help="Teach force attended coefficients with ground truth during model training.")
parser.add_argument(
    "--teach_force_evaluation", default=False, action="store_true",
    help="Teach force attended coefficientes with ground truth during model evaluation.")
parser.add_argument(
    "--preserve_dumps", default=False, action="store_true",
    help="Preserve all the dump files instead of deleting them.")
parser.add_argument(
    "--task_threshold", default=0.05, type=float,
    help="Error threshold required to complete a task.")

# Model parameters
parser.add_argument(
    "--embeddings_size", default=100, type=int,
    help="Size of the word embeddings.")
parser.add_argument(
    "--output_inner_size", default=50, type=int,
    help="Size of the output module inner representations.")
parser.add_argument(
    "--n_blocks", default=20, type=int,
    help="Number of blocks in the dynamic memory.")
parser.add_argument(
    "--temporal_attention_to_sentence", default=False, action="store_true",
    help="Whether to attend or not to the input sentence in temporal attention.")
parser.add_argument(
    "--temporal_activation", default="softmax", choices=("sigmoid", "softmax"),
    help="Which function should the model activate the temporal alignments.")
parser.add_argument(
    "--output_module", default="joint", choices=("joint", "parallel"),
    help="Which output module should the model use.")

# Comet parameters
parser.add_argument(
    "--comet_api_key",
    help="API key to use for Comet.ml")
parser.add_argument(
    "--comet_project_name", default="thesis", type=str,
    help="Project name to use for Comet.ml")
parser.add_argument(
    "--comet_workspace", default="rpalma", type=str,
    help="Workspace to use for Comet.ml")


def run(args):
    # Print the args
    print("Args:", args, "\n")

    # Load the task
    train_split_dataset = BabiTaskDataset(
        dataset_folder_path=args.dataset_folder_path,
        task_id=args.task_id,
        jointly_preprocessed=args.jointly_preprocessed,
        split=BabiTaskDataset.TRAIN)
    metadata = train_split_dataset.metadata
    train_split_size = len(train_split_dataset)
    validation_size = int(args.validation_ratio * train_split_size)
    train_size = train_split_size - validation_size
    train_dataset, validation_dataset = torch.utils.data.dataset.random_split(
        train_split_dataset, [train_size, validation_size])
    test_dataset = BabiTaskDataset(
        dataset_folder_path=args.dataset_folder_path,
        task_id=args.task_id,
        jointly_preprocessed=args.jointly_preprocessed,
        split=BabiTaskDataset.TEST)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False)
    print("Metadata:", metadata, "\n")

    # Experiment identifier and path
    experiment_identifier = time.strftime("%Y%m%d-%H%M%S") + "-qa%d" % args.task_id
    experiment_path = os.path.join(args.checkpoints_folder_path, experiment_identifier)
    os.makedirs(experiment_path)
    print("Experiment path:", experiment_path)

    results = []
    for i in range(1, args.runs + 1):
        # test_loss, test_qa_loss, test_supp_facts_loss, test_accuracy, test_f1, best_epoch
        _, _, _, cycle_accuracy, cycle_f1, cycle_best_epoch = run_cycle(
            args,
            metadata,
            train_dataloader,
            train_split_dataset,
            validation_dataloader,
            test_dataloader,
            experiment_path,
            i
        )
        results.append((i, cycle_accuracy, cycle_f1, cycle_best_epoch))
    
    accuracies = [x[1] for x in results]
    mean_accur = np.mean(accuracies)
    stddev_accur = np.std(accuracies)
    f1s = [x[2] for x in results]
    mean_f1 = np.mean(f1s)
    stddev_f1 = np.std(f1s)
    best_accur_run, best_accur, best_accur_f1, best_accur_epoch = max(results, key=lambda x: x[1])
    best_f1_run, best_f1_accur, best_f1, best_f1_epoch = max(results, key=lambda x: x[2])
    print("Best epoch: %d.%d;  Best F1 epoch: %d.%d" % (best_accur_run, best_accur_epoch, best_f1_run, best_f1_epoch))
    print("Best accuracy: %.5f;  F1: %.5f" % (best_accur, best_accur_f1))
    print("Best F1: %.5f;  accuracy: %.5f" % (best_f1, best_f1_accur))
    print("Mean accuracy: %.5f;  Std. dev. accuracy: %.5f;  Mean F1: %.5f;  Std. dev. F1: %.5f" % (mean_accur, stddev_accur, mean_f1, stddev_f1))

    results_path = os.path.join(experiment_path, "results.log")
    contents = ",".join((
        "task_id",
        "best_accur", "best_error", "best_accur_f1", "mean_accur", "stddev_accur", "best_accur_run", "best_accur_epoch",
        "best_f1", "best_f1_accur", "mean_f1", "stddev_f1", "best_f1_run", "best_f1_epoch")) + "\n"
    contents += ",".join(map(str, (
        args.task_id,
        best_accur, 1 - best_accur, best_accur_f1, mean_accur, stddev_accur, best_accur_run, best_accur_epoch,
        best_f1, best_f1_accur, mean_f1, stddev_f1, best_f1_run, best_f1_epoch))) + "\n"
    with open(results_path, "w") as results_file:
        results_file.write(contents)
    
    if not args.preserve_dumps:
        prune_dump_files(experiment_path, (
            (best_accur_run, best_accur_epoch),
            (best_f1_run, best_f1_epoch)))
        prune_summaries_files(experiment_path, (
            best_accur_run, best_f1_run))


def run_cycle(
        args,
        metadata,
        train_dataloader,
        train_dataset,
        validation_dataloader,
        test_dataloader,
        experiment_path,
        run_id
    ):

    # Comet
    experiment = Experiment(
        api_key=args.comet_api_key,
        project_name=args.comet_project_name,
        workspace=args.comet_workspace)
    experiment.log_parameters(vars(args))
    experiment.log_other("run_id", run_id)

    # Build the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "\n")
    entnet = EntityNetwork(
        embeddings_size=args.embeddings_size,
        vocab_size=metadata["vocab_size"],
        answers_vocab_size=metadata["answers_vocab_size"],
        sentences_length=metadata["max_sentence_length"],
        queries_length=metadata["max_query_length"],
        n_blocks=args.n_blocks,
        output_module=args.output_module,
        output_inner_size=args.output_inner_size,
        temporal_attention_to_sentence=args.temporal_attention_to_sentence,
        temporal_activation=args.temporal_activation,
        dropout_prob=args.dropout_prob,
        device=device)
    entnet.to(device)
    print("Trainable parameters:", sum(p.numel() for p in entnet.parameters() if p.requires_grad), "\n")
    print("Output module:", entnet.output_module, "\n")

    # Set up the loss and optimizer
    qa_criterion = nn.CrossEntropyLoss()
    supp_facts_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([metadata["neg_pos_ratio"]]).to(device))
    optimizer = torch.optim.Adam(
        entnet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)
    schedulers = {
        "step": torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.decay_period,
            gamma=args.decay_rate),
        "cyclical": CyclicalLRScheduler(
            optimizer,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            cycle_length=args.cycle_period)
    }
    scheduler = schedulers[args.lr_scheduler]
    print("Scheduler:", scheduler, "\n")
    optimizer.zero_grad()

    # Build the writers
    train_writer = SummaryWriter(os.path.join(experiment_path, "train-%d" % run_id))
    val_writer = SummaryWriter(os.path.join(experiment_path, "validation-%d" % run_id))
    test_writer = SummaryWriter(os.path.join(experiment_path, "test-%d" % run_id))

    def run_epoch(
            dataloader,
            should_train,
            should_teach_force,
            summaries_writer,
            experiment,
            experiment_context,
            epoch,
            quiet
        ):
        losses = []
        qa_losses = []
        qa_targets = []
        qa_predictions = []
        supp_facts_losses = []
        supp_facts_targets = []
        supp_facts_predictions = []
        entnet.train(mode=should_train)
        for batch in tqdm(dataloader) if not quiet else dataloader:
            story = batch["story"].to(device)
            query = batch["query"].to(device)
            qa_target = batch["answer"].to(device)
            supp_facts_target = batch["supporting"].float().to(device)
            story_mask = batch["story_mask"].float().to(device)
            qa_predicted, supp_facts_alignment, supp_facts_attention = entnet(
                story,
                story_mask,
                query,
                supporting_facts=supp_facts_target if should_teach_force else None)
            qa_loss = qa_criterion(qa_predicted, qa_target)
            supp_facts_loss = supp_facts_criterion(supp_facts_alignment, supp_facts_target)
            loss = args.qa_lambda * qa_loss + args.supporting_facts_lambda * supp_facts_loss
            if should_train:
                loss.backward()
                nn.utils.clip_grad_norm_(entnet.parameters(), args.gradient_clipping)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            qa_losses.append(qa_loss.item())
            qa_targets.append(qa_target.tolist())
            qa_predictions.append(qa_predicted.argmax(dim=1).tolist())
            supp_facts_losses.append(supp_facts_loss.item())
            supp_facts_targets.append(supp_facts_target.tolist())
            supp_facts_predictions.append(supp_facts_attention.tolist())

        if should_train:
            translated_story, translated_query, translated_answer = train_dataset.translate_story(
                story[-1], query[-1], qa_target[-1])
            print("\nSTORY:", translated_story)
            print("QUERY:", translated_query)
            print("ANSWER:", translated_answer)
            print("\nSupporting facts:", supp_facts_target[-1, :])
            print("Attended:", supp_facts_attention[-1, :], "\n")

        mean_loss = np.mean(losses)
        mean_qa_loss = np.mean(qa_losses)
        mean_supp_facts_loss = np.mean(supp_facts_losses)
        mean_qa_accuracy = accuracy(qa_targets, qa_predictions)
        mean_supp_facts_f1 = f1(supp_facts_targets, supp_facts_predictions)

        # Escribir summaries
        write_summaries(
            mean_loss, mean_qa_loss, mean_supp_facts_loss,
            mean_qa_accuracy, mean_supp_facts_f1,
            supp_facts_targets, supp_facts_predictions,
            entnet.named_parameters(),
            summaries_writer, epoch)
        
        with experiment_context():
            metrics = {
                "loss": mean_loss,
                "qa_loss": mean_qa_loss,
                "supp_facts_loss": mean_supp_facts_loss,
                "qa_accuracy": mean_qa_accuracy,
                "supp_facts_f1": mean_supp_facts_f1
            }
            experiment.log_metrics(metrics, step=epoch)
            experiment.log_epoch_end(args.epochs, step=epoch)

        return mean_loss, mean_qa_loss, mean_supp_facts_loss, mean_qa_accuracy, mean_supp_facts_f1
    
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        # Training epoch
        train_loss, train_qa_loss, train_supp_facts_loss, train_qa_accuracy, train_supp_facts_f1 = run_epoch(
            train_dataloader,
            should_train=True,
            should_teach_force=args.teach_force_training,
            summaries_writer=train_writer,
            experiment=experiment,
            experiment_context=experiment.train,
            epoch=epoch,
            quiet=False)
        print("Epoch = %d.%d; task_id = %d\n\ttrain QA accuracy = %.5f; train QA error = %.5f; train supp. facts F1 = %.5f; train loss = %.5f; train QA loss = %.5f; train supp. facts loss = %.5f" % (
                run_id, epoch, args.task_id, train_qa_accuracy, 1 - train_qa_accuracy, train_supp_facts_f1, train_loss, train_qa_loss, train_supp_facts_loss))

        # Validation
        with torch.no_grad():
            val_loss, _, _, val_accuracy, val_f1 = run_epoch(
                validation_dataloader,
                should_train=False,
                should_teach_force=args.teach_force_evaluation,
                summaries_writer=val_writer,
                experiment=experiment,
                experiment_context=experiment.validate,
                epoch=epoch,
                quiet=True)
        print("\tval QA accuracy = %.5f;  val QA error = %.5f;  val loss = %.8f;  val F1 = %.8f" % (val_accuracy, 1 - val_accuracy, val_loss, val_f1), "\n")
        save_model(entnet.state_dict(), experiment_path, run_id, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        # Update learning rate
        scheduler.step()

    # Model evaluation
    entnet.load_state_dict(
        load_model(experiment_path, run_id, best_epoch))
    with torch.no_grad():
        test_loss, test_qa_loss, test_supp_facts_loss, test_accuracy, test_f1 = run_epoch(
            test_dataloader,
            should_train=False,
            should_teach_force=args.teach_force_evaluation,
            summaries_writer=test_writer,
            experiment=experiment,
            experiment_context=experiment.test,
            epoch=best_epoch,
            quiet=False)
    print("Epoch = %d.%d\n\ttest accuracy = %.5f;  test error = %.5f;  test loss = %.8f;  test F1 = %.8f" % (run_id, best_epoch, test_accuracy, 1 - test_accuracy, test_loss, test_f1), "\n")

    return test_loss, test_qa_loss, test_supp_facts_loss, test_accuracy, test_f1, best_epoch

if __name__ == "__main__":
    PARSED_ARGS = parser.parse_args()
    run(PARSED_ARGS)
