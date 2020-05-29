import os
from collections import namedtuple

from tqdm import tqdm
from transformers import BartTokenizer
from typing import List, NamedTuple
from util import data_io

from danqi_chen import danqi_concatenation, fix_brackets


class Turn(NamedTuple):
    request: str
    response: str = None


def get_limited_history(history: List, k: int, hist_len: int):
    return history[max(0, k - hist_len) : (k + 1)]


def coqa(file_name, hist_len=3):

    file = os.environ["HOME"] + "/data/QA/coqa/" + file_name
    data = data_io.read_json(file)["data"]

    def get_history(l: List, k):
        return [
            fix_brackets(d["input_text"]) for d in get_limited_history(l, k, hist_len)
        ]

    for datum in data:
        dialogue_len = len(datum["questions"])
        for k in range(dialogue_len):
            q_hist = get_history(datum["questions"], k)
            a_hist = get_history(datum["answers"], k)
            turns = [Turn(req, res) for req, res in zip(q_hist, a_hist)]
            yield build_input_target(
                fix_brackets(datum["story"]), turns, SEP, use_danqi=True
            )


SILENCE = "<SILENCE>"


def topicalchat(
    file_name="train.json",
    data_path=os.environ["HOME"]
    + "/DIALOGUE/alexa-prize-topical-chat-dataset/conversations",
    hist_len=3,
):

    file = os.path.join(data_path, file_name)
    data = list(data_io.read_json(file).values())
    Utt = namedtuple(
        "Utterance",
        "message agent sentiment knowledge_source turn_rating",
        defaults=[SILENCE] + [None] * 4,
    )

    def build_turn(req: Utt, res: Utt):
        assert req.agent != res.agent
        return Turn(req.message, res.message)

    def build_dialogues(utts):
        turns = [
            build_turn(utts[k], utts[k + 1])
            for k in range(0, len(utterances) // 2 * 2, 2)
        ]
        background = ""
        for k in range(len(turns)):
            some_turns = get_limited_history(turns, k, hist_len)
            yield build_input_target(background, some_turns, SEP)

    for datum in data:
        utterances = [Utt(**d) for d in datum["content"]]
        yield from build_dialogues(utterances)
        # insert silence utter to switch roles
        yield from build_dialogues([Utt()] + utterances)


def squad20(file_name):

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


def personachat(data_path=os.environ["HOME"] + "/data/QA", hist_len=3):
    file_name = "personachat_self_original.json"
    file = os.path.join(data_path, file_name)
    data = data_io.read_json(file)["train"]

    def build_dialogues(background, utt):
        num_utt = len(utt)
        assert num_utt % 2 == 0
        turns = [
            Turn(request=utt[k], response=utt[k + 1]) for k in range(0, num_utt, 2)
        ]
        some_turns = turns[-hist_len:]
        yield build_input_target(background, some_turns, SEP)

    for datum in data:
        background = " ".join(datum["personality"])
        for d in datum["utterances"]:
            response = d["candidates"][-1]
            yield from build_dialogues(background, d["history"] + [response])
            yield from build_dialogues(background, [SILENCE] + d["history"])


def process_text(s):
    return fix_brackets(s.replace("\n", ""))


def build_input_target(background, turns: List[Turn], SEP_TOKEN, use_danqi=False):

    question, target = turns.pop(-1)
    # question = process_text(question) #?
    dialogue = build_input(background, turns, SEP_TOKEN, question, use_danqi)
    return dialogue, target


def build_input(background, turns, SEP_TOKEN, question, use_danqi):
    turns = [(process_text(q), process_text(a)) for q, a in turns]
    if use_danqi:
        dialogue = danqi_concatenation(
            process_text(background), turns, process_text(question)
        )
    else:
        utterances = [x for (q, a) in turns for x in [q, a] if x != SILENCE]
        dialogue = SEP_TOKEN.join([process_text(background)] + utterances + [question])
    return dialogue


tokenizer = BartTokenizer.from_pretrained("bart-large")
# BOS = tokenizer.special_tokens_map['bos_token']
SEP = tokenizer.special_tokens_map["sep_token"]

if __name__ == "__main__":
    datagenerators = {
        "train": [
            # ("topicalchat-train", topicalchat(hist_len=3)),
            # ("personachat-train", personachat(hist_len=3)),
            ("coqa-train", coqa("coqa-train-v1.0.json", hist_len=3)),
            # ("squad20-train", squad20("train-v2.0.json")),
        ],
        "val": [
            ("coqa-val", coqa("coqa-dev-v1.0.json", hist_len=3)),
            # ("squad20-val", squad20("dev-v2.0.json")),
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
"""
corpora-sizes

topicalchat-train: 182347
personachat-train: 262875
coqa-train: 108646
squad20-train: 86820
coqa-val: 7982
squad20-val: 20301
"""
