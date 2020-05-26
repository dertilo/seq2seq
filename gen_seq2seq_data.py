"""
    based on https://github.com/stanfordnlp/coqa-baselines/blob/master/scripts/gen_seq2seq_data.py
"""
from typing import Tuple, List

import argparse
import json
import os
import time


def convert_PTB_tokens_to_normal_ones(s: str):
    return (
        s.replace("-lrb-", "(")
        .replace("-rrb-", ")")
        .replace("-lsb-", "[")
        .replace("-rsb-", "]")
        .replace("-lcb-", "{")
        .replace("-rcb-", "}")
    )


def danqi_concatenation(
    context: str, history: List[Tuple[str, str]], question: str, num_history: int
):
    full_str = context + " ||"
    if num_history < 0:
        for i, (q, a) in enumerate(history):
            d = len(history) - i
            full_str += " <Q{}> ".format(d) + q + " <A{}> ".format(d) + a
    elif num_history > 0:
        context_len = min(num_history, len(history))
        for i, (q, a) in enumerate(history[-context_len:]):
            d = context_len - i
            full_str += " <Q{}> ".format(d) + q + " <A{}> ".format(d) + a
    full_str += " <Q> " + question
    return full_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        "-d",
        type=str,
        default=os.environ["HOME"] + "/data/QA/coqa/coqa-train-v1.0.json",
    )
    parser.add_argument(
        "--n_history",
        type=int,
        default=3,
        help="leverage the previous n_history rounds of Q/A pairs"
        "if n_history == -1, use all history",
    )
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--output_file", "-o", type=str, default="coqa-danqi")
    args = parser.parse_args()

    f_src = open("{}-src.txt".format(args.output_file), "w")
    f_tgt = open("{}-tgt.txt".format(args.output_file), "w")

    with open(args.data_file) as f:
        dataset = json.load(f)

    start_time = time.time()
    data = []
    for idx, datum in enumerate(dataset["data"]):
        if idx % 10 == 0:
            print(
                "processing %d / %d (used_time = %.2fs)..."
                % (idx, len(dataset["data"]), time.time() - start_time)
            )
        context_str = convert_PTB_tokens_to_normal_ones(datum["story"])
        assert len(datum["questions"]) == len(datum["answers"])

        history = []
        for question, answer in zip(datum["questions"], datum["answers"]):
            assert question["turn_id"] == answer["turn_id"]
            idx = question["turn_id"]
            question_str = convert_PTB_tokens_to_normal_ones(question["input_text"])
            answer_str = convert_PTB_tokens_to_normal_ones(answer["input_text"])

            full_str = danqi_concatenation(
                context_str, history, question_str, args.n_history
            )
            if args.lower:
                full_str = full_str.lower()
                answer_str = answer_str.lower()
            f_src.write(full_str + "\n")
            f_tgt.write(answer_str + "\n")
            history.append((question_str, answer_str))

    f_src.close()
    f_tgt.close()
