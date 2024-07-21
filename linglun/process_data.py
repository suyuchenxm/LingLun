import importlib
from linglun.data.preprocess import *
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator, split_midis_for_training
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
from omegaconf import DictConfig


def train_remi_tokenizer(cfg):
    # train REMI tokenizer with BPE
    REMI_TOKENIZER_CONFIG = TokenizerConfig(
        **cfg.tokenizer
    )
    tokenizer = REMI(REMI_TOKENIZER_CONFIG)
    midi_paths = list(Path(cfg.data.datapath).rglob("*.mid"))
    print(f"Training REMI tokenizer with {len(midi_paths)} midi files")
    tokenizer.train(vocab_size=cfg.vocab_size, files_paths=midi_paths)
    tokenizer.save_params(Path("tokenizer.json"))
    print("Tokenizer saved")
    return tokenizer


@hydra.main(config_path=".", config_name="config")
def run_tokenizer(cfg: DictConfig, tokenizer="REMI"):
    DATAPATH = cfg.data.datapath
    FILES = os.listdir(DATAPATH)

    source_path = DATAPATH
    save_path = cfg.data.save_path
    pretrained_model = cfg.data.pretrained_model

    if tokenizer == 'music_transformer':
        pass
    elif tokenizer == "REMI":
        print("Training REMI tokenizer")
        cfg.tokenizer = "remi_default"
        print(cfg.tokenizer)
    elif tokenizer == "pretrained":
        if not pretrained_model:
            raise ValueError("pretrained_model must be specified")
        pass
    else:
        raise ValueError(f"Tokenizer {tokenizer} not found")

if __name__ == "__main__":
    run_tokenizer()
    