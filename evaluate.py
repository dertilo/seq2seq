import os
from pprint import pprint

import torch
from rouge import Rouge
from seq2seq.run_eval import chunks
from seq2seq.utils import use_task_specific_params
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, \
    AutoModelForSeq2SeqLM, AutoTokenizer
from util import data_io

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
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # update config with summarization specific params
    use_task_specific_params(model, "summarization")

    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]
        batch = tokenizer.batch_encode_plus(
            batch, return_tensors="pt", truncation=True, pad_to_max_length=True
        ).to(device)
        summaries = model.generate(**batch, **gen_kwargs)
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            yield hypothesis

if __name__ == "__main__":
    HOME = os.environ["HOME"]
    rouge = Rouge()
    source_file = HOME + "/data/seq2seq_dialogue/val.source"
    sources = [
        " "  # beginning with space? see: https://github.com/huggingface/transformers/blob/5ddd8d6531c8c49fdd281b55b93f6c81c9826f4b/examples/summarization/bart/evaluate_cnn.py#L66
        + x.rstrip()
        for x in data_io.read_lines(
            source_file, limit=1000
        )
    ]
    target_file = HOME + "/data/seq2seq_dialogue/val.target"
    targets = list(
        data_io.read_lines(target_file, limit=1000)
    )
    hyps = list(
        generate_summaries_or_translations(
            sources,
            HOME + "/data/bart_seq2seq_dialogue_continued/checkpointepoch=2.ckpt",
            batch_size=8,fp16=True
        )
    )

    scores = rouge.get_scores(hyps, targets, avg=True)

    pprint(scores)
