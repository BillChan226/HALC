import argparse
import os, sys
import random
sys.path.append("./MiniGPT-4")
sys.path.append("./MiniGPT-4/DoLa")
sys.path.append("./MiniGPT-4/DoLa/transformers-4.28.1")
sys.path.append("./MiniGPT-4/DoLa/transformers-4.28.1/src")
sys.path.append("./MiniGPT-4/DoLa/transformers-4.28.1/src/transformers")
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt
import json
from detector import Detector
from types import SimpleNamespace

# initialize detector
args_dict = {
    'detector_config':"/data/xyq/bill/MiniGPT-4/woodpecker/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    'detector_model_path':"/data/xyq/bill/MiniGPT-4/woodpecker/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    'cache_dir': './cache_dir',
}

class code2_assistant:
    def __init__(self, tokenizer):

        model_args = SimpleNamespace(**args_dict)
        self.detector = Detector(model_args)

        self.tokenizer = tokenizer
        token_vocab_dir = "/data/xyq/bill/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8/tokenizer.json"
        with open(token_vocab_dir, "r") as f:
            self.token_vocab = json.load(f)

        self.token_vocab = self.token_vocab["model"]["vocab"]

        self.token_vocab = {value: key for key, value in self.token_vocab.items()}

    def check_word_complete(self, input_ids):
        input_ids = input_ids[0]
        input_ids = input_ids.cpu().numpy().tolist()

        # print("input_ids", input_ids)
        decoded_tokens = [
            self.tokenizer.decode([token_id]) for token_id in input_ids
        ]
        
        final_tokens = self.token_vocab[input_ids[-1]]
        

        output_text = self.tokenizer.decode(
            input_ids, skip_special_tokens=True
        )

        print("decoded_tokens", decoded_tokens)
        print("output_text", output_text)

        print("final_tokens: ", final_tokens)
        if "▁" in final_tokens:
            last_word_flag = True
            if len(decoded_tokens) < 2:
                last_word = output_text.split(" ")[-1]
            else:
                last_word = output_text.split(" ")[-2]
        else:
            last_word_flag = False
            last_word = "not completed yet!"
        
        return last_word_flag, last_word

    def context_density_embedding(self, entity, context_window=3):
        # context_window specifies the number of context windows
