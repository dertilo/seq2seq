import os
from pprint import pprint

import torch
from rouge import Rouge
from summarization.bart.finetune import SummarizationTrainer
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer
from util import data_io

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(
    examples: list,
    model_name_or_ckpt: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
):
    """
    based on: transformers/examples/summarization/bart/evaluate_cnn.py
    """

    if model_name_or_ckpt.endswith(".ckpt"):
        model = SummarizationTrainer.load_from_checkpoint(model_name_or_ckpt).model.to(
            device
        )
    else:
        model = BartForConditionalGeneration.from_pretrained(model_name_or_ckpt).to(
            device
        )

    tokenizer = BartTokenizer.from_pretrained("bart-large")

    max_length = 140
    min_length = 55

    for batch in tqdm(list(chunks(examples, batch_size))):
        dct = tokenizer.batch_encode_plus(
            batch, max_length=1024, return_tensors="pt", pad_to_max_length=True
        )
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length
            + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_id,
        )
        for g in summaries:
            yield tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    rouge = Rouge()
    sources = [
        " " + x.rstrip() # beginning with space?
        for x in data_io.read_lines(
            HOME + "/data/seq2seq_dialogue/val.source", limit=1000
        )
    ]
    targets = list(
        data_io.read_lines(HOME + "/data/seq2seq_dialogue/val.target", limit=1000)
    )
    hyps = list(
        generate_summaries(
            sources,
            HOME + "/data/bart_seq2seq_dialogue/checkpointepoch=0.ckpt",
            batch_size=8,
        )
    )

    scores = rouge.get_scores(hyps, targets, avg=True)

    pprint(scores)
