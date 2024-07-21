import importlib
from linglun.data.preprocess import *
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig, MusicTokenizer
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from linglun.exceptions import SCORE_LOADING_EXCEPTION
from symusic import Score
import warnings
from omegaconf import OmegaConf


def train_remi_tokenizer(cfg):
    # train REMI tokenizer with BPE
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    REMI_TOKENIZER_CONFIG = TokenizerConfig(
        **cfg_dict
    )
    tokenizer = REMI(REMI_TOKENIZER_CONFIG)
    midi_paths = list(Path(cfg.data.datapath).rglob("*.midi"))
    print(f"Training REMI tokenizer with {len(midi_paths)} midi files")
    tokenizer.train(vocab_size=cfg.vocab_size, files_paths=midi_paths)
    # TODO: new release will deprecate the save_params 
    # tokenizer.save(out_path=".")
    tokenizer.save_params(out_path=".")
    print("Tokenizer saved")
    return tokenizer


def tokenizing(cfg, tokenizer, verbose=True):
    source_path = Path(cfg.data.datapath)
    save_path = Path(cfg.data.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files_paths = os.listdir(source_path)
    desc = f"Tokenizing music files ({'/'.join(list(save_path.parts[-2:]))})"
    for file_path in tqdm(files_paths, desc=desc):
        file_path = Path(source_path/file_path)
        try:
            # load data as score
            score = Score(file_path)
        except FileNotFoundError:
            if verbose:
                warnings.warn(f"File not found: {file_path}", stacklevel=2)
            continue
        except SCORE_LOADING_EXCEPTION as err:
            if verbose:
                warnings.warn(f"Error loading score: {file_path}", stacklevel=2)
            continue
        # tokenize score
        tokens = tokenizer.encode(score)
        out_path = save_path / f"{file_path.stem}.json"
        # save tokenized score
        tokenizer.save_tokens(tokens, out_path)

@hydra.main(config_path="conf/", config_name="config.yaml", version_base='1.2')
def run_tokenizer(cfg: DictConfig):
    tokenizer_type = cfg.tokenizer.name
    save_path = cfg.data.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if tokenizer_type == 'music_transformer':
        pass
    elif tokenizer_type == "remi_default":
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        REMI_TOKENIZER_CONFIG = TokenizerConfig(**cfg_dict)
        tokenizer = REMI(REMI_TOKENIZER_CONFIG)
        print("Using default REMI tokenizer")
        tokenizing(cfg, tokenizer)
        print("dataset tokenized")
    elif tokenizer_type == "remi_custom":
        print("Training REMI tokenizer")
        print(cfg.tokenizer.name)
        tokenizer = train_remi_tokenizer(cfg)
        tokenizing(cfg, tokenizer)
        print("dataset tokenized")
    elif tokenizer_type == "pretrained":
        model_dir = cfg.tokenizer.model_dir
        print(f"Using pretrained tokenizer {model_dir}")
        tokenizer = MusicTokenizer.from_pretrained(model_dir)
        tokenizing(cfg, tokenizer)
        print("dataset tokenized")
    else:
        raise ValueError(f"Tokenizer {tokenizer} not found")

if __name__ == "__main__":
    run_tokenizer()
    