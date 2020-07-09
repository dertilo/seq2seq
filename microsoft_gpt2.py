import os
from pprint import pprint
from typing import List

from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import torch
from util import data_io, util_methods

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("USING: %s" % DEFAULT_DEVICE)


def build_gpt2_input(utterances: List[str]):
    # assert all([isinstance(s,str) for s in utterances])
    # pprint({k:u for k,u in enumerate(utterances)})

    utts = [
        tokenizer.encode(u + tokenizer.eos_token, return_tensors="pt")
        for u in utterances
    ]

    return torch.cat(utts, dim=-1).to(DEFAULT_DEVICE)

    # batch = [[u + tokenizer.eos_token for u in utterances] for utterances in dialogues]
    #
    # return tokenizer.batch_encode_plus(
    #     batch, return_tensors="pt", truncation=True, padding="max_length",
    # ).to(DEFAULT_DEVICE)


def topicalchat(
    file_name="train",
    data_path=os.environ["HOME"] + "/data/QA/topical-chat/processed_output",
    limit=None,
):
    backgrounds = data_io.read_lines(
        os.path.join(data_path, file_name) + ".fct", limit=limit
    )
    dialogs = data_io.read_lines(
        os.path.join(data_path, file_name) + ".src", limit=limit
    )
    targets = data_io.read_lines(
        os.path.join(data_path, file_name) + ".tgt", limit=limit
    )
    for b, d, t in tqdm(zip(backgrounds, dialogs, targets)):
        turns = d.split("_eos")[:-1] + [t.strip("_go").strip("_eos")]
        yield turns[-3:]


def dialogue_test():
    """
    User 	Does money buy happiness?
    Bot 	Depends how much money you spend on it .
    User 	What is the best way to buy happiness ?
    Bot 	You just have to be a millionaire by your early 20s, then you can be happy .
    User 	This is so difficult !
    Bot 	You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money
    """
    user_inputs = [
        "Does money buy happiness?",
        "What is the best way to buy happiness ?",
        "This is so difficult !",
    ]
    # Let's chat for 5 lines
    for step, user_input in enumerate(user_inputs):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        # user_input = input(">> User:")
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, return_tensors="pt"
        )

        # append the new user input tokens to the chat history
        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            if step > 0
            else new_user_input_ids
        )

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )
        # print(tokenizer.decode(chat_history_ids))
        # pretty print last ouput tokens from bot
        print(chat_history_ids.shape)
        print(
            "DialoGPT: {}".format(
                tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                    skip_special_tokens=False,
                )
            )
        )


"""
install transformers from 28th of April, cause microsofts GPT was committed this day
pip install git+https://github.com/huggingface/transformers.git@d714dfeaa8f019a634f2d565fc161f9b17fe85fb
"""

if __name__ == "__main__":


    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = (
        GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-large")
        .to(DEFAULT_DEVICE)
        .eval()
    )

    def answer(input):
        # print(input.shape)
        with torch.no_grad():
            chat_history_ids = model.generate(
                input, max_length=1000, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                # num_beams=3
            )

        output = tokenizer.decode(chat_history_ids[:, input.shape[-1]:][0],
                                  skip_special_tokens=True, )
        # print("OUTPUT: %s"%output)
        return output


    file_name = "valid_freq"
    dialogues_g = topicalchat(
        file_name=file_name,
        data_path=os.environ["HOME"]
        + "/Response-Generation-Baselines/processed_output",
        limit=None
    )
    g = (answer(build_gpt2_input(utts)) for utts in dialogues_g)
    data_io.write_lines("microsoft-gpt2-%s.pred"%file_name, g)

    # dialogue_test()
