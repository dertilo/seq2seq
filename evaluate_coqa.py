from pprint import pprint

import os
from tqdm import tqdm
from util import data_io

from coqa_evaluation import CoQAEvaluator
from seq2seq_chatbot import ChatBot


def test_evaluation(evaluator, data):
    scores = {}
    # gold answers
    pred_data = {
        (datum["id"], q["turn_id"]): a["input_text"]
        for datum in data
        for q, a in zip(datum["questions"], datum["answers"])
    }
    performance = evaluator.model_performance(pred_data)
    scores["gold-answers"] = performance["overall"]
    # questions as answers
    pred_data = {
        (datum["id"], q["turn_id"]): q["input_text"]
        for datum in data
        for q, a in zip(datum["questions"], datum["answers"])
    }
    performance = evaluator.model_performance(pred_data)
    scores["question-answers"] = performance["overall"]
    pprint(scores)
    """
    # TODO(tilo): why are gold-answers not reaching 100% ??
    {'gold-answers': {'em': 94.7, 'f1': 97.3, 'turns': 7983},
     'question-answers': {'em': 0.0, 'f1': 3.5, 'turns': 7983}}
    """


if __name__ == "__main__":
    data_file = os.environ["HOME"] + "/data/QA/coqa/coqa-dev-v1.0.json"
    evaluator = CoQAEvaluator(data_file)

    data = data_io.read_json(data_file)["data"]

    file = "checkpointepoch=2.ckpt"
    checkpoint = os.environ["HOME"] + "/data/bart_coqa_seq2seq/" + file

    with ChatBot(checkpoint) as chatbot:

        def one_dialogue(datum):
            chatbot.reset()
            pred_data = {
                (datum["id"], q["turn_id"]): chatbot.do_answer(
                    q["input_text"], datum["story"]
                )
                for q, a in zip(datum["questions"], datum["answers"])
            }
            return pred_data

        pred_data = {k: a for datum in tqdm(data) for k, a in one_dialogue(datum)}
    performance = evaluator.model_performance(pred_data)
    pprint(performance)

    # test_evaluation(evaluator, data)
