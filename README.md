# Temporal Attention Modules

This repository contains the models and experiments developed for my Master's degree thesis.

```
usage: run.py [-h] [--dataset_folder_path DATASET_FOLDER_PATH]
              [--task_id TASK_ID]
              [--jointly_preprocessed JOINTLY_PREPROCESSED] [--epochs EPOCHS]
              [--batch_size BATCH_SIZE] [--lr_scheduler {step,cyclical}]
              [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE]
              [--decay_period DECAY_PERIOD] [--min_lr MIN_LR]
              [--max_lr MAX_LR] [--cycle_period CYCLE_PERIOD]
              [--gradient_clipping GRADIENT_CLIPPING]
              [--validation_ratio VALIDATION_RATIO]
              [--checkpoints_folder_path CHECKPOINTS_FOLDER_PATH]
              [--runs RUNS] [--dropout_prob DROPOUT_PROB]
              [--qa_lambda QA_LAMBDA]
              [--supporting_facts_lambda SUPPORTING_FACTS_LAMBDA]
              [--weight_decay WEIGHT_DECAY] [--f1_threshold F1_THRESHOLD]
              [--teach_force_training] [--teach_force_evaluation]
              [--preserve_dumps] [--task_threshold TASK_THRESHOLD]
              [--comet_logging COMET_LOGGING] [--teach_force_answer_training]
              [--teach_force_answer_evaluation]
              [--embeddings_size EMBEDDINGS_SIZE]
              [--output_inner_size OUTPUT_INNER_SIZE] [--n_blocks N_BLOCKS]
              [--temporal_attention_to_sentence]
              [--temporal_activation {sigmoid,softmax}]
              [--temporal_attention {additive,multiplicative}]
              [--output_module {joint,parallel}]
              [--temporal_attention_module {prehoc,posthoc}]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_folder_path DATASET_FOLDER_PATH
                        Path to the folder containing the preprocessed
                        dataset.
  --task_id TASK_ID     Task identifier of the bAbI task to run.
  --jointly_preprocessed JOINTLY_PREPROCESSED
                        Whether the task was jointly preprocessed or not.
  --epochs EPOCHS       Number of epochs to train.
  --batch_size BATCH_SIZE
                        Size of the training batches.
  --lr_scheduler {step,cyclical}
                        Learning rate scheduler.
  --learning_rate LEARNING_RATE
                        Learning rate of the training.
  --decay_rate DECAY_RATE
                        How much to decay the learning rate between decay
                        epochs.
  --decay_period DECAY_PERIOD
                        Epochs between learning rate decayment.
  --min_lr MIN_LR       Minimum learning rate for the cyclical scheduling.
  --max_lr MAX_LR       Maximum learning rate for the cyclical scheduling.
  --cycle_period CYCLE_PERIOD
                        Length of a cycle in epochs.
  --gradient_clipping GRADIENT_CLIPPING
                        Upper bound for the gradients.
  --validation_ratio VALIDATION_RATIO
                        Validation set ratio.
  --checkpoints_folder_path CHECKPOINTS_FOLDER_PATH
                        Path to the folder where the checkpoints are going to
                        be stored.
  --runs RUNS           Number of times that the training is repeated.
  --dropout_prob DROPOUT_PROB
                        Probability of dropping out elements in the input
                        module while training.
  --qa_lambda QA_LAMBDA
                        Weight of the question answering loss.
  --supporting_facts_lambda SUPPORTING_FACTS_LAMBDA
                        Weight of the supporting facts loss.
  --weight_decay WEIGHT_DECAY
                        Adam optimizer weight decay.
  --f1_threshold F1_THRESHOLD
                        F1 score threshold
  --teach_force_training
                        Teach force attended coefficients with ground truth
                        during model training.
  --teach_force_evaluation
                        Teach force attended coefficientes with ground truth
                        during model evaluation.
  --preserve_dumps      Preserve all the dump files instead of deleting them.
  --task_threshold TASK_THRESHOLD
                        Error threshold required to complete a task.
  --comet_logging COMET_LOGGING
                        Log experiment to Comet.ml
  --teach_force_answer_training
                        Teach force the answer during post hoc module
                        training.
  --teach_force_answer_evaluation
                        Teach force the answer during post hoc module
                        evaluation.
  --embeddings_size EMBEDDINGS_SIZE
                        Size of the word embeddings.
  --output_inner_size OUTPUT_INNER_SIZE
                        Size of the output module inner representations.
  --n_blocks N_BLOCKS   Number of blocks in the dynamic memory.
  --temporal_attention_to_sentence
                        Whether to attend or not to the input sentence in
                        temporal attention.
  --temporal_activation {sigmoid,softmax}
                        Which function should the model activate the temporal
                        alignments.
  --temporal_attention {additive,multiplicative}
                        Which temporal attention should the model use.
  --output_module {joint,parallel}
                        Which output module should the model use.
  --temporal_attention_module {prehoc,posthoc}
                        Which temporal attention module should the model use.
```