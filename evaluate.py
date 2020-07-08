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

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_summaries_or_translations(
    examples: list,
    model_name_or_ckpt: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    **gen_kwargs,
) -> None:
    """
    based on: transformers/examples/seq2seq/run_eval.py
    """
    if model_name_or_ckpt.endswith(".ckpt"):
        checkpoint = SummarizationModule.load_from_checkpoint(model_name_or_ckpt)
        tokenizer = checkpoint.tokenizer
        model = checkpoint.model.to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_ckpt).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_ckpt)

    if fp16:
        model = model.half()

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


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    rouge = Rouge()
    source_file = HOME + "/data/seq2seq_dialogue/val.source"
    sources = [
        " "  # beginning with space? see: https://github.com/huggingface/transformers/blob/5ddd8d6531c8c49fdd281b55b93f6c81c9826f4b/examples/summarization/bart/evaluate_cnn.py#L66
        + x.rstrip()
        for x in data_io.read_lines(source_file, limit=100)
    ]
    target_file = HOME + "/data/seq2seq_dialogue/val.target"
    model_file = "coqa-distilbart-xsum-12-1/val_avg_rouge2=0.1955-step_count=24.ckpt"
    hyps = list(
        generate_summaries_or_translations(
            sources, model_file, batch_size=8, fp16=True,
        )
    )

    targets = list(data_io.read_lines(target_file, limit=100))

    scores = rouge.get_scores(hyps, targets, avg=True)

    pprint({s+"-f1":v for s,d in scores.items() for k,v in d.items() if k=="f"})

    # bleu_scores = corpus_bleu(hyps, targets).score
    #
    # pprint(bleu_scores)
