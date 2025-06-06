from dataset import BilingualDataset,casual_mask
from model import build_transformer

from config import get_config,latest_weights_file_path,get_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import random_split,DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
from pathlib import Path

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw =  load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config["lang_src"]) 
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"]) 

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])  

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"]) 
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"]) 

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config,vocab_src_len,vocab_tgt_len):
    model =  build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],d_model=config["d_model"])
    return model

def train_model(config):
    # 1) setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2) ensure our checkpoint folder exists (e.g. "opus_books_weights/")
    model_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 3) load data & vocab
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # 4) build & move model
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    # 5) tensorboard + optimizer
    writer    = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 6) default to starting from epoch 0
    initial_epoch = 0
    global_step   = 0

    # 7) optionally preload a checkpoint
    if config.get('preload'):
        if config['preload'] == "latest":
            ckpt_path = latest_weights_file_path(config)
        else:
            ckpt_path = get_weights_file_path(config, config['preload'])

        if ckpt_path and Path(ckpt_path).exists():
            print(f"Preloading checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)

            initial_epoch = state['epoch'] + 1
            global_step   = state.get('global_step', 0)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
        else:
            print("No checkpoint found; starting from scratch.")

    # 8) loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # 9) training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in loop:
            enc_in = batch['encoder_input'].to(device)
            dec_in = batch['decoder_input'].to(device)
            enc_m  = batch['encoder_mask'].to(device)
            dec_m  = batch['decoder_mask'].to(device)
            labels = batch['label'].to(device)

            enc_out = model.encode(enc_in, enc_m)
            dec_out = model.decode(enc_out, enc_m, dec_in, dec_m)
            logits  = model.project(dec_out)

            loss = loss_fn(
                logits.view(-1, tokenizer_tgt.get_vocab_size()),
                labels.view(-1)
            )

            loop.set_postfix(loss=loss.item())
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # 10) save checkpoint at end of each epoch
        ckpt_file = model_dir / f"{config['model_basename']}{epoch:02d}.pt"
        torch.save({
            'epoch':                   epoch,
            'global_step':             global_step,
            'model_state_dict':        model.state_dict(),
            'optimizer_state_dict':    optimizer.state_dict()
        }, ckpt_file)

    writer.close()


if __name__ == '__main__':
    config = get_config()
    train_model(config)




