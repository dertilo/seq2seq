import os

from tqdm import tqdm
from transformers import BartTokenizer
from typing import List, NamedTuple
from util import data_io


class Turn(NamedTuple):
    request: str
    response: str


def generate_coqa_seq2seq(file_name, hist_len=3):

    file = os.environ["HOME"] + "/data/QA/coqa/" + file_name
    data = data_io.read_json(file)["data"]

    def get_limited_history(l: List, k, hist_len):
        return [d["input_text"] for d in l[max(0, k - hist_len) : (k + 1)]]

    for datum in data:
        dialogue_len = len(datum["questions"])
        for k in range(dialogue_len):
            q_hist = get_limited_history(datum["questions"], k, hist_len)
            a_hist = get_limited_history(datum["answers"], k, hist_len)
            turns = [Turn(req, res) for req, res in zip(q_hist, a_hist)]
            yield build_input_target(datum["story"], turns, SEP)


def generate_squad20_seq2seq(file_name):

    file = os.environ["HOME"] + "/data/QA/SQUAD20/" + file_name
    data = data_io.read_json(file)["data"]
    for datum in data:
        for p in datum["paragraphs"]:
            background = p["context"]
            for qa in p["qas"]:
                if not qa["is_impossible"]:
                    q = qa["question"]
                    for a in qa["answers"]:
                        turns = [Turn(q, a["text"])]
                        yield build_input_target(background, turns, SEP)


def generate_personachat_seq2seq(data_path=os.environ["HOME"] + "/data/QA"):
    file_name = "personachat_self_original.json"
    file = os.path.join(data_path, file_name)
    data = data_io.read_json(file)["train"]
    for datum in data:
        background = " ".join(datum["personality"])
        for d in datum["utterances"]:
            last_candidate_is_the_right_one = d["candidates"][-1]
            utterances = d["history"] + [last_candidate_is_the_right_one]
            turns = [
                Turn(request=utterances[k], response=utterances[k + 1])
                for k in range(0, len(utterances), 2)
            ]
            yield build_input_target(background, turns, SEP)


def build_input_target(background, turns: List[Turn], SEP_TOKEN):
    def process(s):
        return s.replace("\n", "")

    turns = [process(x) for turn in turns for x in turn]
    target = process(turns.pop(-1))
    dialogue = SEP_TOKEN.join([process(background)] + turns)
    return dialogue, target


tokenizer = BartTokenizer.from_pretrained("bart-large")
# BOS = tokenizer.special_tokens_map['bos_token']
SEP = tokenizer.special_tokens_map["sep_token"]

if __name__ == "__main__":
    datagenerators = {
        "train": [
            ("personachat-train", generate_personachat_seq2seq()),
            ("coqa-train", generate_coqa_seq2seq("coqa-train-v1.0.json")),
            ("squad20-train", generate_squad20_seq2seq("train-v2.0.json")),
        ],
        "val": [
            ("coqa-val", generate_coqa_seq2seq("coqa-dev-v1.0.json")),
            ("squad20-val", generate_squad20_seq2seq("dev-v2.0.json")),
        ],
    }
    data_path = os.environ["HOME"] + "/data/seq2seq_dialogue"
    os.makedirs(data_path, exist_ok=True)

    for ds, gs in datagenerators.items():
        with open(data_path + "/" + ds + ".source", mode="w") as s, open(
            data_path + "/" + ds + ".target", mode="w"
        ) as t:
            for name, g in gs:
                for k, (x, y) in enumerate(g):
                    s.write(x + "\n")
                    t.write(y + "\n")
