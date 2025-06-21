import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import MDataset


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "output3/checkpoint-9000",
        torch_dtype=torch.float16,
        device_map="auto",
    )


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        "output3/checkpoint-9000",
        use_fast=True,
    )


class Chat:
    def __init__(self):
        self.chat = MDataset.SYSTEM_START_PROMT

    def add_message(self, role: str, message: str) -> None:
        self.chat += MDataset.START_HEADER_TOKEN + \
            role + MDataset.END_HEADER_TOKEN + \
            message + MDataset.EOT_TOKEN

    def new_chat_text(self, text: str) -> None:
        self.chat = text.replace(MDataset.EOT_TOKEN, "") + MDataset.EOT_TOKEN

    def __repr__(self):
        import re
        text = self.chat.replace(MDataset.BEGIN_TOKEN, "").replace(
            MDataset.EOT_TOKEN, "")
        roles = re.findall(
            r'{}(.*?){}'.format(MDataset.START_HEADER_TOKEN.replace("|", "\|"), MDataset.END_HEADER_TOKEN.replace("|", "\|")), text)
        contents = re.split(r'{}.*?{}'.format(MDataset.START_HEADER_TOKEN.replace("|",
                                                                                  "\|"), MDataset.END_HEADER_TOKEN.replace("|", "\|")), text)[1:]

        res = ""
        for role, content in zip(roles, contents):
            res += f"{role}: \n\t{content}\n"
        return res


if __name__ == "__main__":
    model, tokenizer = load_model(), load_tokenizer()

    chat = Chat()
    chat.add_message(role="user", message=input("Message: "))
    while True:
        inputs = tokenizer(chat.chat, return_tensors="pt").to("cuda")
        chat.new_chat_text(tokenizer.batch_decode(
            model.generate(**inputs, max_new_tokens=512))[0])
        print(chat)
        h_input = input("Message: ")
        if h_input == "":
            break
        chat.add_message(role="user", message=h_input)
