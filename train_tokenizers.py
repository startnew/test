from pathlib import Path
import pandas as pd
import os
def train_tok():

    p = "./data/text.txt"
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer()
    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
    tokenizer.train([p], min_frequency=2, special_tokens=special_tokens)
    tokenizer.add_special_tokens(special_tokens)
    with open(p,"r",encoding="utf-8") as f:
        sample = f.readline()
        #break
    #sample = info_pd.description[0]
    strs = sample#" ".join([str(x) for x in sample])
    print(strs)
    encoded = tokenizer.encode(strs)
    print("print(encoded.ids)", encoded.ids)
    print("print(encoded.tokens)", encoded.tokens)
    tok_p = "./user_data/tokenizer"
    os.makedirs(tok_p,exist_ok=True)
    if os.path.exists(os.path.join(tok_p,"vocab.txt")):
        print("tokenizer already exists ")
        pass
    else:
        tokenizer.save_model("./user_data/tokenizer")
    print("get_vocab_size", tokenizer.get_vocab_size())
    return tokenizer
if __name__ == "__main__":
    train_tok()






