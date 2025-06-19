import os
from datasets import load_dataset, load_from_disk, Dataset
from sklearn.model_selection import train_test_split


class MDataset:
    BEGIN_TOKEN: str = "<|begin_of_text|>"
    START_PROMT: str = "You are a coach assistant who must help the user solve his problems using your knowledge."
    EOT_TOKEN: str = "<|eot_id|>"
    START_HEADER_TOKEN: str = "<|start_header_id|>"
    END_HEADER_TOKEN: str = "<|end_header_id|>\n\n"
    SYSTEM_NAME: str = "system"
    ASSISTANT_NAME: str = "assistant"
    USER_NAME: str = "user"
    SYSTEM_START_PROMT: str = BEGIN_TOKEN + START_HEADER_TOKEN + \
        SYSTEM_NAME + END_HEADER_TOKEN + START_PROMT + EOT_TOKEN

    DATASETS: dict[str, str] = {
        "calvindelima/life_coaching_conversations": "life_coaching_conversations",
        "drublackberry/hbr-coaching-real-leaders": "hbr_coaching_real_leaders",
        "asbjoernrubek/clinical_nutritional_coaching": "clinical_nutritional_coaching"
    }

    def __init__(self, disabled_datasets: list[int], limit_of_samples: int) -> None:
        self.data = []
        self.limit_of_samples = limit_of_samples

        for dataset_idx, (dataset_name, func) in enumerate(MDataset.DATASETS.items()):
            if dataset_idx in disabled_datasets:
                continue
            self.__getattribute__(func)(dataset_idx, dataset_name)

    def chats_to_text_and_append(self, chats: list[dict[str, str]]) -> int:
        statr_number_of_samples = len(self.data)
        for chat in chats:
            text = MDataset.SYSTEM_START_PROMT
            for replic in chat:
                if replic["role"] == "system":
                    continue
                text += MDataset.START_HEADER_TOKEN + \
                    replic["role"] + MDataset.END_HEADER_TOKEN + \
                    replic["content"] + MDataset.EOT_TOKEN
            if len(self.data) < self.limit_of_samples:
                self.data.append(text)
        return len(self.data) - statr_number_of_samples

    @staticmethod
    def load_dataset(idx: int, dataset_name: str, split: str | None):
        if split is None:
            if os.path.exists(f"datasets/{idx}/train") and os.path.exists(f"datasets/{idx}/test"):
                dataset = load_from_disk(f"datasets/{idx}")
            else:
                dataset = load_dataset(dataset_name)
                dataset.save_to_disk(f"datasets/{idx}")
            return dataset
        else:
            if os.path.exists(f"datasets/{idx}/{split}"):
                dataset = load_from_disk(f"datasets/{idx}")
            else:
                dataset = load_dataset(dataset_name)
                dataset.save_to_disk(f"datasets/{idx}")
            return dataset

    def life_coaching_conversations(self, idx: int, dataset_name: str) -> None:
        dataset = MDataset.load_dataset(idx, dataset_name, None)
        number_of_samples = 0
        number_of_samples += self.chats_to_text_and_append(
            dataset["train"]["messages"])
        snumber_of_samples += self.chats_to_text_and_append(
            dataset["test"]["messages"])
        print(
            f"Loaded {number_of_samples} samples from `{dataset_name}")

    def hbr_coaching_real_leaders(self, idx: int, dataset_name: str) -> None:
        dataset = MDataset.load_dataset(idx, dataset_name, "train")
        number_of_samples = self.chats_to_text_and_append(
            dataset['train']["messages"])
        print(
            f"Loaded {number_of_samples} samples from `{dataset_name}")

    def clinical_nutritional_coaching(self, idx: int, dataset_name: str) -> None:
        import re
        dataset = MDataset.load_dataset(idx, dataset_name, "train")
        chats = []
        for conv in dataset["train"]["conversation"]:
            if conv.count("Client:") == 0 or conv.count("Nutritional Coach:") == 0:
                continue
            conv = conv.replace("Client:", "|").replace(
                "Nutritional Coach:", "_").replace("[/INST]", "").replace("[INST]", "").replace("<s>", "")
            chat = []
            number_of_chat = 0
            splitted_conv = re.split(r"[|_]", conv)
            for replica in splitted_conv:
                if replica == "":
                    continue
                if number_of_chat % 2 == 0:
                    chat.append(
                        {"role": "user", "content": replica.replace("\n", "")})
                else:
                    chat.append(
                        {"role": "assistant", "content": replica.replace("\n", "")})
                number_of_chat += 1
            chats.append(chat)

        number_of_samples = self.chats_to_text_and_append(chats)
        print(
            f"Loaded {number_of_samples} samples from `{dataset_name}")

    def split_to_train_test(self) -> Dataset:
        train, test = train_test_split(
            self.data, test_size=0.1, random_state=42)
        return Dataset.from_dict({"text": train}), Dataset.from_dict({"text": test})


if __name__ == "__main__":
    d = MDataset()
    print(d.split_to_train_test())
