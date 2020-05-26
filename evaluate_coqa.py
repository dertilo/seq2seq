from pprint import pprint

import os
from util import data_io

from coqa_evaluation import CoQAEvaluator

if __name__ == "__main__":
    data_file = os.environ["HOME"] + "/data/QA/coqa/coqa-dev-v1.0.json"
    evaluator = CoQAEvaluator(data_file)

    data = data_io.read_json(data_file)["data"]
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
