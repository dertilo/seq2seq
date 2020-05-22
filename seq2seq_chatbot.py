import os

import torch
from summarization.bart.finetune import SummarizationTrainer
from tqdm import tqdm
from transformers import BartTokenizer
from util import data_io
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import TEXT, ID, Schema

from build_seq2seq_corpus import build_input_target, Turn

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_answer(
    SEP, background, device, max_length, min_length, model, tokenizer, utt
):
    inputt, _ = build_input_target(background, [Turn(utt,"nix")], SEP)
    batch = [" " + inputt]
    dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt")
    encoded = model.generate(
        input_ids=dct["input_ids"].to(device),
        attention_mask=dct["attention_mask"].to(device),
        num_beams=4,
        length_penalty=2.0,
        max_length=max_length + 2,
        # +2 from original because we start at step=1 and stop before max_length
        min_length=min_length + 1,  # +1 from original because we start at step=1
        no_repeat_ngram_size=3,
        early_stopping=True,
        decoder_start_token_id=model.config.eos_token_id,
    )[0]
    answer = tokenizer.decode(
        encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return answer


def run_interaction(model_name: str, device: str = DEFAULT_DEVICE):

    assert model_name.endswith(".ckpt")
    model = SummarizationTrainer.load_from_checkpoint(model_name).model.to(device)
    tokenizer = BartTokenizer.from_pretrained("bart-large")
    SEP = tokenizer.special_tokens_map["sep_token"]

    max_length = 140
    min_length = 10

    from whoosh.qparser import QueryParser

    ix = index.open_dir(INDEX_DIR)
    qp = QueryParser("story", schema=ix.schema)

    with ix.searcher() as s:
        while True:
            utt = input(": ")
            or_searche = " OR ".join(utt.split(" "))
            q = qp.parse(or_searche)
            results = s.search(q, limit=1)
            if len(results) > 0:
                background = results[0]["story"]
                answer = generate_answer(
                    SEP,
                    background,
                    device,
                    max_length,
                    min_length,
                    model,
                    tokenizer,
                    utt,
                )
            else:
                answer = "Whaat?"
            print(answer)


def build_index(data, schema, index_dir="indexdir"):

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    ix = index.create_in(index_dir, schema)

    writer = ix.writer()
    for d in tqdm(data):
        writer.add_document(**d)
    writer.commit()


def build_schema_and_corpus():
    schema = Schema(
        id=ID(stored=True),
        filename=ID(stored=True),
        story=TEXT(analyzer=StemmingAnalyzer(), stored=True, lang="en"),
    )
    file = os.environ["HOME"] + "/data/QA/coqa/" + "coqa-train-v1.0.json"
    data = (
        {"id": d["id"], "filename": d["filename"], "story": d["story"]}
        for d in data_io.read_json(file)["data"]
    )
    return schema, data


INDEX_DIR = "coqa_index"
if not os.path.isdir(INDEX_DIR):
    schema, data = build_schema_and_corpus()
    build_index(data, schema, index_dir=INDEX_DIR)
    print("done building corpus")


if __name__ == "__main__":
    model_file = os.environ["HOME"] + "/data/bart_seq2seq_dialogue/checkpointepoch=0.ckpt"
    run_interaction(model_file)