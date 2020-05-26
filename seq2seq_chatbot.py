import os
from typing import List

import spacy
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
    SEP,
    background,
    max_length,
    min_length,
    model,
    tokenizer,
    dialog_history: List[Turn],
):
    inputt, _ = build_input_target(background, dialog_history, SEP)
    batch = [" " + inputt]
    dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt")
    encoded = model.generate(
        input_ids=dct["input_ids"].to(DEFAULT_DEVICE),
        attention_mask=dct["attention_mask"].to(DEFAULT_DEVICE),
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


from whoosh.qparser import QueryParser


class ChatBot:

    max_length = 140
    min_length = 10

    def __init__(self, checkpoint_file) -> None:
        assert checkpoint_file.endswith(".ckpt")
        self.model = SummarizationTrainer.load_from_checkpoint(
            checkpoint_file
        ).model.to(DEFAULT_DEVICE)
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        self.SEP = self.tokenizer.special_tokens_map["sep_token"]
        self.spacy_nlp = spacy.load("en_core_web_sm")
        super().__init__()

    def __enter__(self):
        ix = index.open_dir(INDEX_DIR)
        self.searcher = ix.searcher()
        self.qp = QueryParser("story", schema=ix.schema)
        self.background = (
            "The weather was rainy today, but maybe its going to be sunny tomorrow."
        )
        self.dialogue_history: List[Turn] = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.searcher.__exit__(exc_type, exc_val, exc_tb)

    def respond(self, utt: str):
        self.dialogue_history.append(Turn(utt, "nix"))
        doc = self.spacy_nlp(utt)
        entities = [s.text for s in doc.ents]
        if len(entities) > 0:
            self._update_background(entities)

        answer = generate_answer(
            self.SEP,
            self.background,
            self.max_length,
            self.min_length,
            self.model,
            self.tokenizer,
            self.dialogue_history[-2:],
        )
        self.dialogue_history[-1] = Turn(utt, answer)
        return answer, self.background

    def _update_background(self, entities):
        or_searche = " OR ".join(entities)
        q = self.qp.parse(or_searche)
        results = self.searcher.search(q, limit=1)
        if len(results) > 0:
            self.background = results[0]["story"]


def run_interaction(checkpoint: str):
    with ChatBot(checkpoint) as chatbot:
        while True:
            utt = input(": ")
            if utt == "bye":
                print("bye")
                break
            else:
                respond, background = chatbot.respond(utt)
                print(respond)


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
    file = "checkpointepoch=2.ckpt"
    model_file = os.environ["HOME"] + "/data/bart_coqa_seq2seq/" + file
    run_interaction(model_file)
