import argparse
import os
from pprint import pprint
from typing import List, Dict

import torch
from rouge import Rouge
from sacrebleu import corpus_bleu
from seq2seq.finetune import SummarizationModule
from seq2seq.run_eval import chunks
from seq2seq.utils import use_task_specific_params
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
)
from util import data_io

from evaluation import calc_rouge_scores

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_summaries_or_translations(
    examples: list,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    **gen_kwargs,
) -> None:
    """
    based on: transformers/examples/seq2seq/run_eval.py
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # update config with summarization specific params
    use_task_specific_params(model, "summarization")

    for batch in tqdm(list(chunks(examples, batch_size))):

        dec = batch_generate(batch, model, tokenizer, gen_kwargs, device)
        for hypothesis in dec:
            yield hypothesis


def batch_generate(
    batch: List,
    model: BartForConditionalGeneration,
    tokenizer,
    gen_kwargs: Dict,
    device: str = DEFAULT_DEVICE,
) -> List:
    batch_dict = tokenizer.batch_encode_plus(
        batch, return_tensors="pt", truncation=True, pad_to_max_length=True
    ).to(device)
    summaries = model.generate(**batch_dict, **gen_kwargs)
    dec = tokenizer.batch_decode(
        summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return dec


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    default="topicalchat-distilbart-xsum-12-1/best_tfmr",
    type=str,
)
parser.add_argument(
    "--source_file",
    default=os.environ["HOME"] + "/data/QA/topical-chat/processed_output/test_rare.src",
    type=str,
)
parser.add_argument(
    "--target_file",
    default=os.environ["HOME"] + "/data/QA/topical-chat/processed_output/test_rare.tgt",
    type=str,
)
parser.add_argument(
    "--pred_file",
    default="test_rare.pred",
    type=str,
)

if __name__ == "__main__":
    args = parser.parse_args()

    rouge = Rouge()
    sources = [
        " "  # beginning with space? see: https://github.com/huggingface/transformers/blob/5ddd8d6531c8c49fdd281b55b93f6c81c9826f4b/examples/summarization/bart/evaluate_cnn.py#L66
        + x.rstrip()
        for x in data_io.read_lines(args.source_file)
    ]
    targets = list(data_io.read_lines(args.target_file))
    hyps = list(
        generate_summaries_or_translations(
            sources, args.model_path, batch_size=8, fp16=True,
        )
    )
    data_io.write_lines(args.pred_file,hyps)

    pprint(calc_rouge_scores(hyps,targets))
