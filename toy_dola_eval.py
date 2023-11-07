# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py

import re
import os
import json
import random
import torch
import numpy as np

# define pre-trained model download path by setting the environment variable
os.environ["TRANSFORMERS_CACHE"] = "./model_checkpoints/"
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request

from DoLa.dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True


def build_prompt(input_text, n_shot, cot_flag, shuffle):
    input_text_prompt = "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=23)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./gsm8k")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    toy_text = "Who was the first Nigerian to win the Nobel Prize, in which year?"

    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(",")]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
    elif len(early_exit_layers) == 2:
        print(
            f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}"
        )
        mode = "dola-static"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(
            f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}"
        )
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l: 0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    answers = []
    result_dict = {
        "is_correct": [],
        "model_answer": [],
        "model_completion": [],
        "full_input_text": [],
    }
    # for sample in tqdm(list_data_dict):
    input_text = build_prompt(toy_text, N_SHOT, COT_FLAG, args.do_shuffle)
    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        mode=mode,
        mature_layer=mature_layer,
        premature_layer=premature_layer,
        candidate_premature_layers=candidate_premature_layers,
        relative_top=args.relative_top,
    )
    model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
    if mode == "dola":
        for k, v in c_dist.items():
            premature_layer_dist[k] += v

    result_dict["model_completion"].append(model_completion)
    result_dict["full_input_text"].append(input_text)
    if DEBUG:
        print(f"Full input_text:\n{input_text}\n\n")
    print(f"Question: {toy_text}\n\n" f"Model Completion: {model_completion}\n\n")

    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print(
                    "Premature layer {0} was used {1} times, {2}%".format(
                        l,
                        premature_layer_dist[l],
                        round(premature_layer_dist[l] / total_tokens * 100, 2),
                    )
                )
    # save results to a json file
    model_tag = (
        model_name.split("/")[-1]
        if model_name[-1] != "/"
        else model_name.split("/")[-2]
    )
    output_file = (
        args.output_path
        if args.shard_id is None
        else (args.output_path + "_" + str(args.shard_id) + ".json")
    )
