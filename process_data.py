from linglun.data.preprocess import *
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig, MusicTokenizer
# from miditok.pytorch_data import DatasetMIDI, DataCollator
# from torch.utils.data import DataLoader
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from linglun.exceptions import SCORE_LOADING_EXCEPTION
from symusic import Score
import warnings
from omegaconf import OmegaConf
import json


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


def miditok_tokenizing(cfg, tokenizer, verbose=True):
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

def musictransformer_tokenizing(cfg):
    source_path = Path(cfg.data.datapath)
    save_path = Path(cfg.data.save_path)
    files_paths = list(Path(source_path).rglob("*.midi"))
    print(f"Tokenizing {len(files_paths)} music files")
    desc = f"Tokenizing music files ({'/'.join(list(save_path.parts[-2:]))})"
    for file_path in tqdm(files_paths, desc=desc):
        file_path = Path(file_path)
        try:
            data = encode_midi(str(file_path))
        except KeyboardInterrupt:
            print('Aborted')
            return 
        except SCORE_LOADING_EXCEPTION as e: 
            print("Error loading score: ", file_path)
        
        out_path = save_path / f"{file_path.stem}.json"
        with open(out_path, 'w') as f:
            json.dump(data, f)

@hydra.main(config_path="conf/", config_name="config.yaml", version_base='1.2')
def run_tokenizer(cfg: DictConfig):
    tokenizer_type = cfg.tokenizer.name
    os.makedirs(cfg.data.save_path, exist_ok=True)

    if tokenizer_type == 'music_transformer':
        musictransformer_tokenizing(cfg)
        print("dataset tokenized")
    elif tokenizer_type == "remi_default":
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        REMI_TOKENIZER_CONFIG = TokenizerConfig(**cfg_dict)
        tokenizer = REMI(REMI_TOKENIZER_CONFIG)
        print("Using default REMI tokenizer")
        miditok_tokenizing(cfg, tokenizer)
        print("dataset tokenized")
    elif tokenizer_type == "remi_custom":
        print("Training REMI tokenizer")
        print(cfg.tokenizer.name)
        tokenizer = train_remi_tokenizer(cfg)
        miditok_tokenizing(cfg, tokenizer)
        print("dataset tokenized")
    elif tokenizer_type == "pretrained":
        model_dir = cfg.tokenizer.model_dir
        print(f"Using pretrained tokenizer {model_dir}")
        tokenizer = MusicTokenizer.from_pretrained(model_dir)
        miditok_tokenizing(cfg, tokenizer)
        print("dataset tokenized")
    else:
        raise ValueError(f"Tokenizer {tokenizer} not found")

if __name__ == "__main__":
    run_tokenizer()
    