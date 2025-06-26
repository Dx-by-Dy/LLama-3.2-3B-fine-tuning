import argostranslate.package
import argostranslate.translate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BEGIN_TOKEN: str = "<|begin_of_text|>"
START_PROMT: str = "You are a coach assistant who must help the user solve his problems using your knowledge. Be calm like a professional, don't show emotions."
EOT_TOKEN: str = "<|eot_id|>"
START_HEADER_TOKEN: str = "<|start_header_id|>"
END_HEADER_TOKEN: str = "<|end_header_id|>\n\n"
SYSTEM_NAME: str = "system"
ASSISTANT_NAME: str = "assistant"
USER_NAME: str = "user"
SYSTEM_START_PROMT: str = BEGIN_TOKEN + START_HEADER_TOKEN + \
    SYSTEM_NAME + END_HEADER_TOKEN + START_PROMT + EOT_TOKEN


MODEL_PATH: str = "output3/checkpoint-9000"
# OUTPUT: str = "output4/checkpoint-27000"


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
    )


def load_translators(from_lang: str, to_lang: str):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    available_package = list(
        filter(
            lambda x: x.from_code == from_lang and x.to_code == to_lang, available_packages
        )
    )[0]
    download_path = available_package.download()
    argostranslate.package.install_from_path(download_path)

    available_package = list(
        filter(
            lambda x: x.from_code == to_lang and x.to_code == from_lang, available_packages
        )
    )[0]
    download_path = available_package.download()
    argostranslate.package.install_from_path(download_path)

    installed_languages = argostranslate.translate.get_installed_languages()

    from_to_translator = list(filter(
        lambda x: x.code == from_lang,
        installed_languages))[0].get_translation(list(filter(
            lambda x: x.code == to_lang,
            installed_languages))[0])

    to_from_translator = list(filter(
        lambda x: x.code == to_lang,
        installed_languages))[0].get_translation(list(filter(
            lambda x: x.code == from_lang,
            installed_languages))[0])

    return (from_to_translator, to_from_translator)
