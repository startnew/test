
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tokenizers.processors import BertProcessing
from tokenizers import Tokenizer
from train_tokenizers import train_tok
from torch.utils.data import Dataset
import torch
import random


class MyDataset(Dataset):
    def __init__(self, tokenizer, evaluate=False):
        self.examples = []
        src_files = ["./data/text.txt"]

        for src_file in src_files:
            print("?", src_file)
            with open(src_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                self.examples += [x.ids for x in tokenizer.encode_batch(lines)]
        random.seed(1)
        random.shuffle(self.examples)
        nums = int(len(self.examples) * 0.1)
        print("num eval", nums)
        print("num train", len(self.examples) - nums)
        if evaluate:
            self.examples = self.examples[:nums]
        else:
            self.examples = self.examples[nums:]
        self.examples = self.examples
        print(self.examples[0])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


###############train
from transformers import AlbertConfig

if __name__ == "__main__":
    
    name = "./Deberta_v2_base"
    tokenizer = train_tok()
    train_data_path = ""
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=120)

    train_data_set = MyDataset(tokenizer, evaluate=False)
    test_data_set = MyDataset(tokenizer, evaluate=True)
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("./user_data/tokenizer", max_len=120)
    vocab_size = len(tokenizer)
    learning_rate = 1E-4

    from transformers import DebertaV2Config

    config = DebertaV2Config(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.0,
        num_hidden_groups=1,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        down_scale_factor=1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        position_biased_input=False,
        max_relative_positions=-1,
        relative_attention=True,
        pos_att_type="c2p|p2c",
    )

    from transformers import DebertaV2ForMaskedLM

    model = DebertaV2ForMaskedLM(config=config)

 
    print("model_num_parameters resize before", model.num_parameters())
    model.resize_token_embeddings(len(tokenizer))
    print("model_num_parameters", model.num_parameters())
    print(model)
    from transformers import  DataCollatorForWholeWordMask


    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    # )
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    from transformers import Trainer, TrainingArguments

   
    epochs = 100
    per_device_train_batch_size = 10
    print("batch_size is :{}".format(per_device_train_batch_size))
    training_args = TrainingArguments(
        output_dir="./{}".format(name),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=2000,
        save_total_limit=2,
        learning_rate=learning_rate,
        warmup_ratio=1/320,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data_set,
        tokenizer=tokenizer,
        eval_dataset=test_data_set,
    )
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    print(os.path.isdir(
                training_args.output_dir) ,training_args.output_dir)
    if os.path.isdir(
                training_args.output_dir) :
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("resume from :{}".format(last_checkpoint))
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model("./{}".format(name))
