from typing import NamedTuple, Any, Generator, List

import os
from util import data_io


class DialogRequest(NamedTuple):
    dialogue_id: Any
    turn_id: Any
    is_start: bool
    background: str
    question: str

class Answer(NamedTuple):
    dialogue_id: Any
    turn_id: Any
    utterance: str


def utt_generator(dialog_it, get_seq_fun):
    while True:
        try:
            data = next(dialog_it)
            seq = get_seq_fun(data)
            is_start = True
            for d in seq:
                yield is_start, d
                is_start = False
        except StopIteration as e:
            yield None


def coqa_to_batches(data, batch_size=3) -> Generator[List[DialogRequest], None, None]:
    dialog_it = iter(data)

    def get_id_questions(d):
        return [(d["id"], d["story"], q) for q in d["questions"]]

    gs = [utt_generator(dialog_it, get_id_questions) for _ in range(batch_size)]
    while True:
        batch = list(filter(None, [next(g) for g in gs]))
        batch = [
            DialogRequest(
                dialogue_id, q["turn_id"], is_start, background, q["input_text"]
            )
            for is_start, (dialogue_id, background, q) in batch
        ]
        if len(batch) > 0:
            yield batch
        else:
            break


if __name__ == "__main__":
    data_file = os.environ["HOME"] + "/data/QA/coqa/coqa-dev-v1.0.json"

    data = data_io.read_json(data_file)["data"][:5]

    for batch in coqa_to_batches(data):
        print(batch)
