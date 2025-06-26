from support import (ASSISTANT_NAME,
                     END_HEADER_TOKEN,
                     EOT_TOKEN,
                     START_HEADER_TOKEN,
                     START_PROMT,
                     SYSTEM_NAME,
                     SYSTEM_START_PROMT,
                     USER_NAME,
                     load_model,
                     load_tokenizer,
                     load_translators)
import re


class Chat:
    def __init__(self, model_lang: str, user_lang: str, silent: bool = True):
        self.model_to_user_translator, self.user_to_model_translator = load_translators(
            model_lang, user_lang)
        self.model_lang = model_lang
        self.user_lang = user_lang
        self.silent = silent

        self.model = load_model()
        self.model.eval()
        self.tokenizer = load_tokenizer()
        self.model_chat = SYSTEM_START_PROMT
        self.chat: list[dict[str, str]] = [
            {"role": SYSTEM_NAME,
             "content_model_lang": START_PROMT,
             "content_user_lang": self.model_to_user_translator.translate(START_PROMT)}]

        self.print_last_role()
        self.print_last_user_lang_message()
        self.print_last_model_lang_message()

    def dialog(self) -> bool:
        user_input = input(f"\n{USER_NAME}:\n\t")
        if user_input == "":
            return False

        self.add_user_message(user_input)
        self.get_assistant_replic()

        self.print_last_role()
        self.print_last_user_lang_message()
        self.print_last_model_lang_message()
        return True

    def tg_api(self, user_message: str) -> tuple[str, str, str]:
        self.add_user_message(user_message)
        self.get_assistant_replic()
        return self.chat[-2]["content_model_lang"], self.chat[-1]["content_user_lang"], self.chat[-1]["content_model_lang"]

    def get_assistant_replic(self) -> None:
        inputs = self.tokenizer(
            self.model_chat, return_tensors="pt").to("cuda")
        self.add_assistant_message(self.tokenizer.batch_decode(
            self.model.generate(**inputs, max_new_tokens=512))[0])

    def add_user_message(self, message: str) -> None:
        translated_message = self.user_to_model_translator.translate(message)
        self.chat.append(
            {"role": USER_NAME, "content_model_lang": translated_message, "content_user_lang": message})
        self.model_chat += START_HEADER_TOKEN + \
            USER_NAME + END_HEADER_TOKEN + \
            translated_message + EOT_TOKEN
        self.print_last_model_lang_message()

    def add_assistant_message(self, chat: str) -> None:
        content = chat.replace("\n", "").rsplit("{}".format(END_HEADER_TOKEN.replace('\n', '')), 1)[
            1].replace(EOT_TOKEN, "")
        content = re.sub(r"<\|.*?\|>", "", content)
        self.chat.append({"role": ASSISTANT_NAME, "content_model_lang": content,
                         "content_user_lang": self.model_to_user_translator.translate(content)})
        self.model_chat += START_HEADER_TOKEN + \
            ASSISTANT_NAME + END_HEADER_TOKEN + \
            content

    def print_last_role(self):
        if not self.silent:
            print(f"\n{self.chat[-1]['role']}:")

    def print_last_model_lang_message(self):
        if not self.silent:
            print(f"\t{self.chat[-1]['content_model_lang']}")

    def print_last_user_lang_message(self):
        if not self.silent:
            print(f"\t{self.chat[-1]['content_user_lang']}")

    def __repr__(self):
        res = ""
        for replic in self.chat:
            res += f"{replic['role']}:\n\t{replic['content_user_lang']}\n\t{replic['content_model_lang']}\n"
        return res


if __name__ == "__main__":
    chat = Chat(model_lang="en", user_lang="ru")
    while chat.dialog():
        pass
