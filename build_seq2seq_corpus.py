import os

from tqdm import tqdm
from transformers import BartTokenizer
from typing import List
from util import data_io


def generate_coqa_seq2seq(file_name, hist_len=3):

    data = data_io.read_json(os.environ["HOME"] + "/data/QA/coqa/" + file_name)["data"]

    def get_history(l: List, k, hist_len):
        return [d["input_text"] for d in l[max(0, k - hist_len) : (k + 1)]]

    for datum in data:
        dialogue_len = len(datum["questions"])
        for k in range(dialogue_len):
            q_hist = get_history(datum["questions"], k, hist_len)
            a_hist = get_history(datum["answers"], k, hist_len)
            dialogue, target = build_input_target(datum["story"], q_hist, a_hist, SEP)
            yield dialogue, target


def generate_squad20_seq2seq(file_name):

    data = data_io.read_json(os.environ["HOME"] + "/data/QA/SQUAD20/" + file_name)[
        "data"
    ]
    for datum in data:
        for p in datum["paragraphs"]:
            background = p["context"]
            for qa in p["qas"]:
                if not qa["is_impossible"]:
                    q = qa["question"]
                    for a in qa["answers"]:
                        dialogue, target = build_input_target(
                            background, [q], [a["text"]], SEP
                        )
                        yield dialogue, target


def generate_personachat_seq2seq(file_name):

    data = data_io.read_json(os.environ["HOME"] + "/data/QA/" + file_name)["train"]
    for datum in data:
        background = " ".join(datum["personality"])
        for d in datum["utterances"]:
            x = d["history"] + d["candidates"][-1:]
            qas, aas = list(zip(*[x[k : k + 2] for k in range(0, len(x), 2)]))
            dialogue, target = build_input_target(background, qas, aas, SEP)
            yield dialogue, target


def build_input_target(background, q_hist: List[str], a_hist: List[str], SEP_TOKEN):
    def process(s):
        return s.replace("\n", "")

    turns = [process(x) for turn in zip(q_hist, a_hist) for x in turn]
    target = process(turns.pop(-1))
    dialogue = SEP_TOKEN.join([process(background)] + turns)
    return dialogue, target


tokenizer = BartTokenizer.from_pretrained("bart-large")
# BOS = tokenizer.special_tokens_map['bos_token']
SEP = tokenizer.special_tokens_map["sep_token"]

if __name__ == "__main__":
    datagenerators = {
        "train": [
            (
                "personachat-train",
                generate_personachat_seq2seq("personachat_self_original.json"),
            ),
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
