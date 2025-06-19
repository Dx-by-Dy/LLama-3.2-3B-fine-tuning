from peft import LoraConfig
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch


def load_new_model(quantization_mod: int):
    if quantization_mod == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8"
        )

        return AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        )
    if quantization_mod == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        return AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )


def load_new_tokenizer():
    return AutoTokenizer.from_pretrained(
        "unsloth/Llama-3.2-3B-Instruct",
        use_fast=True,
    )


def anable_peft(model):
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",]
    )

    model.add_adapter(peft_config)
    return model
