import os

from tqdm import tqdm
from transformers import BartTokenizer
from typing import List, NamedTuple
from util import data_io


class Turn(NamedTuple):
    request: str
    response: str


def get_limited_history(history: List, k: int, hist_len: int):
    return history[max(0, k - hist_len) : (k + 1)]


def generate_coqa_seq2seq(file_name, hist_len=3):

    file = os.environ["HOME"] + "/data/QA/coqa/" + file_name
    data = data_io.read_json(file)["data"]

    def get_history(l: List, k):
        return [d["input_text"] for d in get_limited_history(l, k, hist_len)]

    for datum in data:
        dialogue_len = len(datum["questions"])
        for k in range(dialogue_len):
            q_hist = get_history(datum["questions"], k)
            a_hist = get_history(datum["answers"], k)
            turns = [Turn(req, res) for req, res in zip(q_hist, a_hist)]
            yield build_input_target(datum["story"], turns, SEP)


def generate_topical_chat_seq2seq(file_name="train.json", hist_len=3):

    file = (
        os.environ["HOME"]
        + "/code/DIALOGUE/alexa-prize-topical-chat-dataset/conversations/"
        + file_name
    )
    data = list(data_io.read_json(file).values())

    def build_turn(req, res):
        assert req["agent"] != res["agent"]
        return Turn(req["message"], res["message"])

    def build_dialogues(utts):
        assert len(utts) % 2 == 0
        turns = [build_turn(utts[k], utts[k + 1]) for k in range(0, len(utts), 2)]
        background = ""
        for k in range(len(turns)):
            some_turns = get_limited_history(turns, k, hist_len)
            yield build_input_target(background, some_turns, SEP)

    for datum in data:
        utterances = datum["content"]
        yield from build_dialogues(utterances[0 : len(utterances) // 2 * 2])
        utterance_switched_roles = utterances[1:]
        yield from build_dialogues(
            utterance_switched_roles[0 : len(utterance_switched_roles) // 2 * 2]
        )


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
            ("topicalchat-train", generate_topical_chat_seq2seq()),
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
                num_samples = k
                print("%s: %d" % (name, num_samples))
