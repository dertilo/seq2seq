

from transformers import AutoModelWithLMHead, AutoTokenizer
import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
# model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

"""
User 	Does money buy happiness?
Bot 	Depends how much money you spend on it .
User 	What is the best way to buy happiness ?
Bot 	You just have to be a millionaire by your early 20s, then you can be happy .
User 	This is so difficult !
Bot 	You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money
"""
user_inputs=["Does money buy happiness?","What is the best way to buy happiness ?","This is so difficult !"]
# Let's chat for 5 lines
for step,user_input in enumerate(user_inputs):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    # user_input = input(">> User:")
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(chat_history_ids))
    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))