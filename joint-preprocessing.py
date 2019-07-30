"""
Preprocessing utilities for the bAbI tasks.
"""
import re
import os
import json
import csv
import argparse

from tqdm import tqdm

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets_folder_path", default="/Users/rodolfo/U/Magister/datasets/")
parser.add_argument(
    "--babi_tasks_mode", default="en", choices=("en", "en-10k"))

# Preproccessing based on https://github.com/jimfleming/recurrent-entity-networks
PAD_TOKEN = "_PAD"
PAD_ID = 0

def tokenize(sentence):
    """
    Maps a sentence to a list of tokens. For example:
        'Mary moved to the bathroom.' => ['mary', 'moved', 'to', 'the', 'bathroom', '.']
        'Where is Mary?' => ['where', 'is', 'mary', '?']
    """
    return [token.strip().lower() for token in re.split(r"\W+", sentence) if token.strip()]

def get_parsed_supporting_facts(raw_supportings, q_ids):
    supporting = []
    for raw_supporting in raw_supportings:
        offset = 0
        for q_id in q_ids:
            if raw_supporting > q_id:
                offset += 1
        supporting.append(raw_supporting - offset)
    return supporting


def parse_stories(file_path, only_supporting=False):
    """
    Parse a bAbI task file. Returns a list with bAbI stories.
    Each bAbI story has the following structure:
        (substory, query, answer) where:
        - substory is a list of facts.
        - query is a list of words.
        - answer is a word.
    If `supporting_facts` is `True`, only the supporting facts
    are included in the substory list.
    """
    with open(file_path, "r") as file_descriptor:
        stories = []
        story = []
        for line in file_descriptor:
            line_id, statement = line.split(" ", maxsplit=1)
            line_id = int(line_id)
            if line_id == 1:
                # Reset the story if the line identifier is 1.
                story = []
                questions_ids = []
            if "\t" in statement:
                # Handle Q&A.
                questions_ids.append(line_id)
                query, answer, supporting_facts = statement.split("\t")
                supporting_facts = list(map(int, supporting_facts.strip().split(" ")))
                query = tokenize(query)
                if only_supporting:
                    # Append a substory including only the previous supporting facts.
                    substory = [story[i - 1] for i in supporting_facts]
                else:
                    # Append a substory with all the facts.
                    substory = [fact for fact in story if fact != ""]
                parsed_supporting_facts = get_parsed_supporting_facts(supporting_facts, questions_ids)
                stories.append((substory, query, answer, parsed_supporting_facts))
                story.append("") # Empty fact as placeholder.
            else:
                # Handle fact.
                statement = tokenize(statement)
                story.append(statement)
    return stories


def truncate_stories(stories, max_length):
    """
    Truncates the length of stories' facts to a maximum length.
    """
    stories_truncated = []
    for story, query, answer, supporting in stories:
        story_length = len(story)
        story_truncated = story[-max_length:]
        if story_length > max_length:
            diff = story_length - max_length
            supporting = [x - diff for x in supporting if x - diff > 0]
        stories_truncated.append((story_truncated, query, answer, supporting))
    return stories_truncated


def get_tokenizer(stories):
    """
    Computes a vocabulary and a token map based on the stories.
    """
    tokens = []
    answers_token = []
    for story, query, answer, _ in stories:
        tokens.extend([token for fact in story for token in fact] + query)
        answers_token.extend([answer])
    vocab = [PAD_TOKEN] + sorted(set(tokens))
    answers_vocab = sorted(set(answers_token))
    token_map = {token: i for i, token in enumerate(vocab)}
    answers_token_map = {token: i for i, token in enumerate(answers_vocab)}
    return vocab, answers_vocab, token_map, answers_token_map


def tokenize_stories(stories, token_map, answers_token_map):
    tokenized_stories = []
    for story, query, answer, supporting in stories:
        story = [[token_map[token] for token in fact] for fact in story]
        query = [token_map[token] for token in query]
        answer = answers_token_map[answer]
        tokenized_stories.append((story, query, answer, supporting))
    return tokenized_stories


def get_max_lengths(stories):
    facts_lengths = [len(fact) for story, _, _, _ in stories for fact in story]
    stories_lengths = [len(story) for story, _, _, _ in stories]
    query_lengths = [len(query) for _, query, _, _ in stories]
    return max(facts_lengths), max(stories_lengths), max(query_lengths)


def pad_stories(stories, max_facts, max_stories, max_query):
    for story, query, _, _ in stories:
        for fact in story:
            for _ in range(max_facts - len(fact)):
                fact.append(PAD_ID)
        for _ in range(max_stories - len(story)):
            story.append([PAD_ID for _ in range(max_facts)])
        for _ in range(max_query - len(query)):
            query.append(PAD_ID)
    return stories


def save_dataset(stories, file_path):
    """
    with tf.python_io.TFRecordWriter(file_path) as tf_writer:
        for story, query, answer in stories:
            story_flat = [token_id for sentence in story for token_id in sentence]

            story_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=story_flat))
            query_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=query))
            answer_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[answer]))

            features = tf.train.Features(feature={
                "story": story_feature,
                "query": query_feature,
                "answer": answer_feature,
            })

            example = tf.train.Example(features=features)
            tf_writer.write(example.SerializeToString())
    """
    with open(file_path, "w") as target_file:
        json.dump(stories, target_file)


def main(args):
    BABI_TASKS_FOLDER_PATH = args.datasets_folder_path + "tasks_1-20_v1-2/%s/" % args.babi_tasks_mode
    BABI_TASKS_TOKENIZED_FOLDER = BABI_TASKS_FOLDER_PATH[:-1] + "_tokenized/"
    BABI_TASKS_NAMES = [
        "qa1_single-supporting-fact",
        "qa2_two-supporting-facts",
        "qa3_three-supporting-facts",
        "qa4_two-arg-relations",
        "qa5_three-arg-relations",
        "qa6_yes-no-questions",
        "qa7_counting",
        "qa8_lists-sets",
        "qa9_simple-negation",
        "qa10_indefinite-knowledge",
        "qa11_basic-coreference",
        "qa12_conjunction",
        "qa13_compound-coreference",
        "qa14_time-reasoning",
        "qa15_basic-deduction",
        "qa16_basic-induction",
        "qa17_positional-reasoning",
        "qa18_size-reasoning",
        "qa19_path-finding",
        "qa20_agents-motivations",
    ]
    BABI_TASKS_TITLES = [
        "Task 1: Single Supporting Fact",
        "Task 2: Two Supporting Facts",
        "Task 3: Three Supporting Facts",
        "Task 4: Two Argument Relations",
        "Task 5: Three Argument Relations",
        "Task 6: Yes/No Questions",
        "Task 7: Counting",
        "Task 8: Lists/Sets",
        "Task 9: Simple Negation",
        "Task 10: IndefiniteKnowledg",
        "Task 11: Basic Coreference",
        "Task 12: Conjunction",
        "Task 13: Compound Coreference",
        "Task 14: Time Reasoning",
        "Task 15: Basic Deduction",
        "Task 16: Basic Induction",
        "Task 17: Positional Reasoning",
        "Task 18: Size Reasoning",
        "Task 19: Path Finding",
        "Task 20: Agent Motivations",
    ]
    BABI_TASKS_IDS = [
        "qa1",
        "qa2",
        "qa3",
        "qa4",
        "qa5",
        "qa6",
        "qa7",
        "qa8",
        "qa9",
        "qa10",
        "qa11",
        "qa12",
        "qa13",
        "qa14",
        "qa15",
        "qa16",
        "qa17",
        "qa18",
        "qa19",
        "qa20",
    ]

    if not os.path.exists(BABI_TASKS_TOKENIZED_FOLDER):
        os.makedirs(BABI_TASKS_TOKENIZED_FOLDER)

    all_stories_train = {}
    all_stories_test = {}
    joint_stories_train = []
    joint_stories_test = []
    for task_id, task_name, task_title in zip(tqdm(BABI_TASKS_IDS), BABI_TASKS_NAMES, BABI_TASKS_TITLES):
        # Get all the paths...
        stories_train_path = os.path.join(BABI_TASKS_FOLDER_PATH, task_name + "_train.txt")
        stories_test_path = os.path.join(BABI_TASKS_FOLDER_PATH, task_name + "_test.txt")

        # The capacity of the memory is limited to the most recent 70 sentences, except for task 3
        # which was limited to 130 sentences.
        truncated_story_length = 130 if task_id == "qa3" else 70

        # Parse stories.
        stories_train = parse_stories(stories_train_path)
        stories_test = parse_stories(stories_test_path)

        # Truncate stories
        stories_train = truncate_stories(stories_train, truncated_story_length)
        stories_test = truncate_stories(stories_test, truncated_story_length)

        # Add to stories
        all_stories_train[task_id] = stories_train
        all_stories_test[task_id] = stories_test

        # Add to joint stories
        joint_stories_train = joint_stories_train + stories_train
        joint_stories_test = joint_stories_test + stories_test

    # Get vocabulary.
    print("Get vocabulary and token_map...")
    vocab, answers_vocab, token_map, answers_token_map = get_tokenizer(joint_stories_train + joint_stories_test)

    # Get the max lengths.
    # print("Get max lengths...")
    max_facts_length, _, max_query_length = get_max_lengths(joint_stories_train + joint_stories_test)

    # Add joint task to stories.
    all_stories_train["qa-1"] = joint_stories_train
    all_stories_test["qa-1"] = joint_stories_test

    print("Tokenizing, padding and dumping!")
    for task_id in tqdm(all_stories_train.keys()):
        stories_train = all_stories_train[task_id]
        stories_test = all_stories_test[task_id]

        # Tokenize.
        stories_train = tokenize_stories(stories_train, token_map, answers_token_map)
        stories_test = tokenize_stories(stories_test, token_map, answers_token_map)

        # Pad the stories, sentences and queries.
        _, max_stories_length, _ = get_max_lengths(stories_train + stories_test)
        max_lengths = (max_facts_length, max_stories_length, max_query_length)
        stories_train = pad_stories(stories_train, *max_lengths)
        stories_test = pad_stories(stories_test, *max_lengths)

        # Get pos/neg ratio
        supp_facts_pos = sum(len(supp_facts) for _, _, _, supp_facts in stories_train)
        supp_facts_neg = len(stories_train) * max_lengths[1] - supp_facts_pos
        neg_pos_ratio = supp_facts_neg / supp_facts_pos

        # Dump the datasets and vocabulary.
        dataset_train_path = os.path.join(BABI_TASKS_TOKENIZED_FOLDER, "%s_joint_train.json" % task_id)
        dataset_test_path = os.path.join(BABI_TASKS_TOKENIZED_FOLDER, "%s_joint_test.json" % task_id)
        metadata_path = os.path.join(BABI_TASKS_TOKENIZED_FOLDER, "%s_joint_metadata.json" % task_id)
        metadata_vocab_path = os.path.join(BABI_TASKS_TOKENIZED_FOLDER, "%s_joint_metadata_vocab.tsv" % task_id)
        save_dataset(stories_train, dataset_train_path)
        save_dataset(stories_test, dataset_test_path)
        with open(metadata_path, "w") as metadata_file:
            metadata = {
                "task_id": task_id,
                "task_name": "JOINT",
                "task_title": "JOINT",
                "max_query_length": max_lengths[2],
                "max_story_length": max_lengths[1],
                "max_sentence_length": max_lengths[0],
                "vocab": vocab,
                "neg_pos_ratio": neg_pos_ratio,
                "vocab_size": len(vocab),
                "answers_vocab": answers_vocab,
                "answers_vocab_size": len(answers_vocab),
                "filenames": {
                    "train": os.path.basename(dataset_train_path),
                    "test": os.path.basename(dataset_test_path)
                }
            }
            json.dump(metadata, metadata_file)
        with open(metadata_vocab_path, "w") as metadata_vocab_file:
            tsv_writer = csv.writer(metadata_vocab_file, delimiter="\t")
            for i, word in enumerate(vocab):
                tsv_writer.writerow([word])

if __name__ == "__main__":
    PARSED_ARGS = parser.parse_args()
    main(PARSED_ARGS)
