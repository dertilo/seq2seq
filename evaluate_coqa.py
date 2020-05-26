from pprint import pprint

import os
from tqdm import tqdm
from util import data_io, util_methods

from coqa_evaluation import CoQAEvaluator
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

    def do_answer(self, story_id, turn_id):
        return self.gold_preds[(story_id, turn_id)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def reset(self):
        pass


def evaluate_chatbot(chatbot,batch_size=1):
    def one_dialogue(datum):
        chatbot.reset()
        g = util_methods.iterable_to_batches(datum["questions"],batch_size=batch_size)
        for batch in g:
            if isinstance(chatbot, CheatBot):
                answers = [chatbot.do_answer(datum["id"], batch[0]["turn_id"])]
            else:
                batch = [(q["input_text"], datum["story"]) for q in batch]
                answers = chatbot.do_answer(batch)
            for q,a in zip(batch,answers):
                yield (datum["id"], q["turn_id"]), a

    g = ((k, a) for datum in data for k, a in one_dialogue(datum))
    pred_data = {k: a for k, a in tqdm(g)}
    performance = evaluator.model_performance(pred_data)
    return performance


if __name__ == "__main__":
    data_file = os.environ["HOME"] + "/data/QA/coqa/coqa-dev-v1.0.json"
    evaluator = CoQAEvaluator(data_file)

    data = data_io.read_json(data_file)["data"]

    file = "checkpointepoch=2.ckpt"
    checkpoint = os.environ["HOME"] + "/data/bart_coqa_seq2seq/" + file

    scores = {}
    scores["cheatbot"] = evaluate_chatbot(CheatBot(data))
    scores["echobot"] = evaluate_chatbot(CheatBot(data, do_echo=True))
    with ChatBot(checkpoint, find_background=False) as chatbot:
        scores["bart"] = evaluate_chatbot(chatbot,batch_size=4)
    pprint({n: s["overall"] for n, s in scores.items()})

    """
    # TODO(tilo): why are gold-answers not reaching 100% ??
    {'cheatbot': {'em': 94.7, 'f1': 97.3, 'turns': 7983},
     'echobot': {'em': 0.0, 'f1': 3.5, 'turns': 7983}}
    """
