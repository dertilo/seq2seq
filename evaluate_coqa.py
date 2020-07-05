import sys
from typing import List

from pprint import pprint

import os
from tqdm import tqdm
from util import data_io

from batchify_dialogues import coqa_to_batches, DialogRequest, Answer
from coqa_evaluator import CoQAEvaluator
from seq2seq_chatbot import ChatBot


class CheatBot:
    def __init__(self, data, do_echo=False) -> None:
        def answer_fun(q, a):
            if do_echo:
                return q["input_text"]
            else:
                return a["input_text"]

        self.gold_preds = {
            (datum["id"], q["turn_id"]): answer_fun(q, a)
            for datum in data
            for q, a in zip(datum["questions"], datum["answers"])
        }
        super().__init__()

    def do_answer(self, batch: List[DialogRequest]) -> List[Answer]:
        return [self.do_cheat(r.dialogue_id, r.turn_id) for r in batch]

    def do_cheat(self, story_id, turn_id) -> Answer:
        return Answer(story_id, turn_id, self.gold_preds[(story_id, turn_id)])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def evaluate_chatbot(chatbot, data, batch_size=1):

    g = (
        ((a.dialogue_id, a.turn_id), a.utterance)
        for batch in coqa_to_batches(data, batch_size)
        for a in chatbot.do_answer(batch)
    )
    pred_data = {k: v for k, v in tqdm(g)}

    performance = evaluator.model_performance(pred_data)
    return performance


if __name__ == "__main__":
    data_file = os.environ["HOME"] + "/data/QA/coqa/coqa-dev-v1.0.json"
    evaluator = CoQAEvaluator(data_file)

    data = data_io.read_json(data_file)["data"]
    scores = {}

    # file = "checkpointepoch=2.ckpt"
    # checkpoint = os.environ["HOME"] + "/data/bart_coqa_seq2seq/" + file

    # scores["cheatbot"] = evaluate_chatbot(CheatBot(data), data)
    # scores["echobot"] = evaluate_chatbot(CheatBot(data, do_echo=True), data)
    # with ChatBot(checkpoint, find_background=False, use_danqi=False) as chatbot:
    #     scores["bart"] = evaluate_chatbot(chatbot, data, batch_size=16)

    checkpoint = sys.argv[1]

    with ChatBot(checkpoint, find_background=False, use_danqi=False) as chatbot:
        scores["bart-danqi"] = evaluate_chatbot(chatbot, data, batch_size=16)
    pprint({n: s["overall"] for n, s in scores.items()})

    """
    # TODO(tilo): why are gold-answers not reaching 100% ??
    {'bart': {'em': 47.9, 'f1': 65.5, 'turns': 7983}}
    {'bart-danqi': {'em': 31.6, 'f1': 43.7, 'turns': 7983}} # TODO(tilo): why is danqi not working?
     'cheatbot': {'em': 94.7, 'f1': 97.3, 'turns': 7983},
     'echobot': {'em': 0.0, 'f1': 3.5, 'turns': 7983}}
    """
