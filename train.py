import argparse
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from dataset import MDataset
from new_model import anable_peft, load_new_model, load_new_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs_train", type=int, default=1,
                        help="Batch size for the train dataset. (default = `1`)")
    parser.add_argument("--bs_eval", type=int, default=1,
                        help="Batch size for the test dataset. (default = `1`)")
    parser.add_argument("--gr_acc", type=int, default=1,
                        help="Gradient accumulation steps. (default = `1`)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate. (default = `2e-5`)")
    parser.add_argument("--num_epoch", type=int, default=1,
                        help="Number of train epochs. (default = `1`)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory. (default = `output`)")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence lenght. (default = `2048`)")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluating model steps frequency. (default = `1000`)")
    parser.add_argument("--log_steps", type=int, default=50,
                        help="Logging trainer steps frequency. (default = `50`)")
    parser.add_argument("--save_steps", type=int, default=250,
                        help="Saving stete steps frequency. (default = `250`)")
    parser.add_argument("--quant", type=int, default=0, choices=[0, 4, 8],
                        help="Quantization type. 0 - without quantization. (default = `0`)")
    parser.add_argument("--lr_type", type=str, default="constant",
                        help="Learning rate scheduler type. (default = `constant`)")
    parser.add_argument("--disabled_datasets", type=int, default=[], nargs='*',
                        help="Disabled datasets with indices. (default = `[]`)")
    parser.add_argument("--limit_samples", type=int, default=0,
                        help="Limit of number samples. 0 - no limit. (default = `0`)")
    args = parser.parse_args()

    model = load_new_model(args.quant)
    tokenizer = load_new_tokenizer()
    model = anable_peft(model)
    train_dataset, test_dataset = MDataset(
        args.disabled_datasets, args.limit_samples).split_to_train_test()

    if args.quant == 8:
        optim = "adamw_torch_8bit"
    elif args.quant == 4:
        optim = "adamw_torch_4bit"
    else:
        optim = "adamw_torch"

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template=MDataset.START_HEADER_TOKEN +
            MDataset.ASSISTANT_NAME + MDataset.END_HEADER_TOKEN,
            instruction_template=MDataset.START_HEADER_TOKEN +
            MDataset.USER_NAME + MDataset.END_HEADER_TOKEN,
            tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=args.bs_train,
            per_device_eval_batch_size=args.bs_eval,
            gradient_accumulation_steps=args.gr_acc,
            learning_rate=args.lr,
            lr_scheduler_type=args.lr_type,
            num_train_epochs=args.num_epoch,
            logging_steps=args.log_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            optim=optim,
            overwrite_output_dir=True,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_len,
        ),
    )

    trainer_stats = trainer.train()

    # python train.py --gr_acc 4 --num_epoch 10 --output_dir output4 --max_seq_len 256 --quant 8 --lr_type linear --disabled_datasets 0 1 --limit_samples 100000
