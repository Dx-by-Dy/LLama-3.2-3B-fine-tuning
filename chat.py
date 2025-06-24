import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import MDataset

# OUTPUT: str = "output3/checkpoint-9000"
OUTPUT: str = "output4/checkpoint-27000"


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        OUTPUT,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        OUTPUT,
        use_fast=True,
    )


class Chat:
    def __init__(self):
        self.model = load_model()
        self.tokenizer = load_tokenizer()
        self.model_chat = MDataset.SYSTEM_START_PROMT
        self.chat: list[dict[str, str]] = [
            {"role": MDataset.SYSTEM_NAME, "content": MDataset.START_PROMT}]

        self.print_last_message()

    def dialog(self) -> bool:
        user_input = input(f"{MDataset.USER_NAME}:\n\t")
        if user_input == "":
            return False

        self.add_user_message(user_input)
        self.get_assistant_replic()
        self.print_last_message()
        return True

    def get_assistant_replic(self) -> None:
        inputs = self.tokenizer(
            self.model_chat, return_tensors="pt").to("cuda")
        self.add_assistant_message(self.tokenizer.batch_decode(
            self.model.generate(**inputs, max_new_tokens=512))[0])

    def add_user_message(self, message: str) -> None:
        self.chat.append({"role": MDataset.USER_NAME, "content": message})
        self.model_chat += MDataset.START_HEADER_TOKEN + \
            MDataset.USER_NAME + MDataset.END_HEADER_TOKEN + \
            message + MDataset.EOT_TOKEN

    def add_assistant_message(self, chat: str) -> None:
        import re

        content = chat.replace("\n", "").rsplit("{}{}{}".format(MDataset.START_HEADER_TOKEN, MDataset.ASSISTANT_NAME, MDataset.END_HEADER_TOKEN.replace('\n', '')), 1)[
            1].replace(MDataset.EOT_TOKEN, "")
        content = re.sub(r"<\|.*?\|>", "", content)
        self.chat.append({"role": MDataset.ASSISTANT_NAME, "content": content})
        self.model_chat += MDataset.START_HEADER_TOKEN + \
            MDataset.ASSISTANT_NAME + MDataset.END_HEADER_TOKEN + \
            content

    def print_last_message(self):
        print(f"\n{self.chat[-1]['role']}:\n\t{self.chat[-1]['content']}\n")

    def __repr__(self):
        # import re
        # text = self.chat.replace(MDataset.BEGIN_TOKEN, "").replace(
        #     MDataset.EOT_TOKEN, "")
        # roles = re.findall(
        #     r'{}(.*?){}'.format(MDataset.START_HEADER_TOKEN.replace("|", "\|"), MDataset.END_HEADER_TOKEN.replace("|", "\|")), text)
        # contents = re.split(r'{}.*?{}'.format(MDataset.START_HEADER_TOKEN.replace("|",
        #                                                                           "\|"), MDataset.END_HEADER_TOKEN.replace("|", "\|")), text)[1:]

        res = ""
        for replic in self.chat:
            res += f"{replic['role']}:\n\t{replic['content']}\n"
        return res


if __name__ == "__main__":
    chat = Chat()
    while chat.dialog():
        pass
