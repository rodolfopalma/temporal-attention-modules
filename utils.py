import os
import glob
import shutil

import numpy as np
import torch


def save_model(state_dict, experiment_path, run_id, epoch):
    path = os.path.join(experiment_path, "%d.%d.pth" % (run_id, epoch))
    torch.save({
        "epoch": epoch,
        "state_dict": state_dict,
    }, path)


def load_model(experiment_path, run_id, best_epoch, map_to_cpu=False):
    path = os.path.join(experiment_path, "%d.%d.pth" % (run_id, best_epoch))
    print("Loading %s..." % path)
    checkpoint = torch.load(path, map_location="cpu") if map_to_cpu else torch.load(path)
    return checkpoint["state_dict"]


def write_summaries(
        loss,
        qa_loss,
        supp_facts_loss,
        qa_accuracy,
        supp_facts_f1,
        supp_facts_targets,
        supp_facts_predictions,
        named_params,
        writer,
        step
    ):
    # Scalars
    writer.add_scalar("loss", loss, step)
    writer.add_scalar("qa_loss", qa_loss, step)
    writer.add_scalar("supp_facts_loss", supp_facts_loss, step)
    writer.add_scalar("qa_accuracy", qa_accuracy, step)
    writer.add_scalar("supp_facts_f1", supp_facts_f1, step)

    # P/R curve
    # [n_batches, batch_sz, story_len]
    targets_list = [el for batch in supp_facts_targets for el in batch]
    targets = np.array(targets_list).flatten()
    predictions_list = [el for batch in supp_facts_predictions for el in batch]
    predictions = np.array(predictions_list).flatten()
    writer.add_pr_curve("supporting_facts", targets, predictions, step)

    # Histograms
    for name, param in named_params:
        writer.add_histogram(name, param.clone().cpu().data.numpy(), step)


def prune_dump_files(experiment_path, epochs):
    pattern = os.path.join(experiment_path, "*.*.pth")
    dumps = glob.glob(pattern)
    best_dump_names = ["%d.%d.pth" % (e[0], e[1]) for e in epochs]
    for dump_path in dumps:
        _, dump_tail = os.path.split(dump_path)
        if dump_tail not in best_dump_names:
            os.remove(dump_path)

def prune_summaries_files(experiment_path, runs):
    pattern = os.path.join(experiment_path, "*-*")
    logdirs = glob.glob(pattern)
    for logdir_path in logdirs:
        _, logdir_name = os.path.split(logdir_path)
        run = int(logdir_name.split("-")[1])
        if run not in runs:
            shutil.rmtree(logdir_path)
