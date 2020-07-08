import argparse
from pprint import pprint

import os
from typing import List

from rouge import Rouge
from seq2seq.utils import calculate_rouge
from util import data_io

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pred_file",
    default=os.environ["HOME"] + "/gunther/data/transformer_trained/test_rare_epoch_20.pred",
    type=str,
)
parser.add_argument(
    "--target_file",
    default=os.environ["HOME"] + "/gunther/Response-Generation-Baselines/processed_output/test_rare.tgt",
    type=str,
)


def calc_rouge_scores(pred:List[str],tgt:List[str]):
    rouge = Rouge()
    scores = rouge.get_scores(pred, tgt, avg=True)
    scores = {
        "f1-scores": {s: v for s, d in scores.items() for k, v in d.items() if
                      k == "f"},
        "huggingface-rouge": calculate_rouge(pred, tgt)
    }
    return scores


if __name__ == "__main__":

    args = parser.parse_args()

    pred = list(data_io.read_lines(args.pred_file))
    tgt = list(data_io.read_lines(args.target_file))

    pprint(calc_rouge_scores(pred,tgt))
