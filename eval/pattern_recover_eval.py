import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms

from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
import json

from types import SimpleNamespace
from decoder_zoo.Woodpecker.vis_corrector import Corrector
from decoder_zoo.Woodpecker.config import woodpecker_args_dict
from decoder_zoo.HALC.context_density.halc import halc_assistant
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
}


def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="minigpt4", help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="greedy",
    help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
)
parser.add_argument(
    "-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="coco",
    help="Name of the dataset. Default is 'coco'.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/czr/HaLC/MME_benchmark/pattern_exp/existence",
    help="data path",
)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("-b", "--beam", type=int, default=3)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", "--num_samples", type=int, default=100)
parser.add_argument("-m", "--max_new_tokens", type=int, default=64)
parser.add_argument(
    "-v",
    "--verbosity",
    action="store_false",
    dest="verbosity",
    default=True,
    help="Verbosity. Default: True.",
)
parser.add_argument(
    "-k",
    "--k-candidate-num",
    type=int,
    default=4,
    help="specify the k candidate number for halc.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./log/",
    help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
)
parser.add_argument(
    "-p",
    "--post-correction",
    type=str,
    default=None,
    help="Post correction method such as woodpecker, lure.",
)
parser.add_argument(
    "-e",
    "--expand-ratio",
    type=float,
    default=0.6,
    help="Expand ratio of growing contextual field.",
)
parser.add_argument(
    "--cd_alpha",
    type=float,
    default=1,
    help="Alpha param for VCD.",
)
parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta param for VCD.")
parser.add_argument("--noise_step", type=int, default=500, help="Noise step for VCD.")
parser.add_argument(
    "--hallucination_caption", type=str, default=None, help="Hallucination caption."
)

args = parser.parse_known_args()[0]

# print("args.gpu_id", args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
model_name = args.model
decoding_strategy = args.decoder
cfg = Config(args)
seed = args.seed
setup_seeds(cfg, seed)

device = (
    torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
)
# device = "cpu"

verbosity = args.verbosity
k_candidate_num = args.k_candidate_num
num_samples = args.num_samples
dataset_name = args.dataset_name
data_path = args.data_path
output_dir = args.output_dir
num_beams = args.beam
num_workers = args.num_workers
batch_size = args.batch_size
post_correction = args.post_correction
max_new_tokens = args.max_new_tokens
expand_ratio = args.expand_ratio
cd_alpha = args.cd_alpha
cd_beta = args.cd_beta
hallucination_caption_path = args.hallucination_caption

loaded_json = []
loaded_keys = []
with open(hallucination_caption_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        pair = json.loads(line)
        if pair["answer"] == False:
            loaded_json.append(pair)
            loaded_keys.append(pair["image_id"])

print("hallucination number: ", len(loaded_keys))
# ========================================
#             Model Initialization
# ========================================
print("Initializing Model")

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
print("model device", model.device)

print("expand_ratio", expand_ratio)
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)

valid_decoding_strategies = [
    "greedy",
    "dola",
    "halc-dola",
    "halc-greedy",
    "halc-beam",
    "opera-beam",
    "vcd",
]
valid_post_editing_strategies = ["lure", "woodpecker"]

assert (
    decoding_strategy in valid_decoding_strategies
), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"
assert (
    post_correction in valid_post_editing_strategies or post_correction is None
), f"Invalid post correction strategy: {post_correction}, should be in {valid_post_editing_strategies}"

decoding_strategy = decoding_strategy
opera_decoding = False
dola_decoding = False
halc_decoding = False
vcd_decoding = False
beam_search = False

print("decoding_strategy", decoding_strategy)
if decoding_strategy == "greedy":
    pass
elif decoding_strategy == "dola":
    dola_decoding = True

elif decoding_strategy == "halc-dola":
    dola_decoding = True
    halc_decoding = True
elif decoding_strategy == "halc-greedy":
    halc_decoding = True
elif decoding_strategy == "halc-beam":
    halc_decoding = True
    dola_decoding = True
    beam_search = True
elif decoding_strategy == "opera-beam":
    beam_search = True
    opera_decoding = True
elif decoding_strategy == "vcd":
    vcd_decoding = True


if post_correction == "woodpecker":
    model_args = SimpleNamespace(**woodpecker_args_dict)
    corrector = Corrector(model_args)


print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} #######\033[0m")


if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("dataset_name: ", dataset_name)
    print("data_path: ", data_path)
    print("output_dir: ", output_dir)
    print("num_samples: ", num_samples)
    print("num_beams: ", num_beams)
    print("seed: ", seed)
    print(vis_processors["eval"].transform)


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


if not os.path.exists(data_path):
    raise ValueError("data_path does not exist!")
else:
    file_names = os.listdir(data_path)

# Initialize an empty dictionary to store the paths
file_dict = {}

# Go through each file name to pair .txt files with corresponding .jpg files
for file_name in file_names:
    if file_name.endswith(".txt"):
        # Construct the full path for the .txt file
        txt_file_path = os.path.join(data_path, file_name)

        # Construct the corresponding .jpg file name
        img_file_name = file_name.replace(".txt", ".jpg")
        img_id = int(img_file_name.split("/")[-1].split(".")[0])
        # Check if the corresponding .jpg file exists in the list
        if img_file_name in file_names and img_id in loaded_keys:
            # Construct the full path for the .jpg file
            jpg_file_path = os.path.join(data_path, img_file_name)

            # Add to the dictionary
            file_dict[txt_file_path] = jpg_file_path

print("file_dict", file_dict)

print("hallucination number", len(file_dict.keys()))


base_dir = output_dir + "/" + args.model
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

halc_params = {
    "context_domain": "upper",
    "contrast_weight": 0.05,
    "context_window": 4,
    "expand_ratio": expand_ratio,
    "beam_size": num_beams,
    "k_candidate_num": args.k_candidate_num,
    "LVLM_backbone": model_name,
}
halc_assistant_helper = halc_assistant(
    model,
    vis_processor=vis_processor,
    device=device,
    halc_params=halc_params,
    max_new_tokens=max_new_tokens,
)

hallucination_list = []

for txt_path in tqdm(file_dict):
    # print("txt_path", txt_path)
    img_path = file_dict[txt_path]

    with open(txt_path, "r") as file:
        text_contents = file.read().strip()

    img_save = {}
    img_save["ground_truth"] = text_contents
    img_save["image_id"] = int(img_path.split("/")[-1].split(".")[0])

    image_path = img_path
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    # print("image device", norm(image).device)

    if "?" in text_contents:
        qu = text_contents.split("?")[0] + "?"
        answer = text_contents.split("?")[-1]
    else:
        qu = text_contents.split(".")[0] + "."
        answer = text_contents.split(".")[-1]

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    lm_early_exit_layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
    ]

    mature_layer = lm_early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = lm_early_exit_layers[:-1]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    halc_assistant_helper.update_input(img_path=image_path, input_prompt=qu)

    image_cd = None

    if vcd_decoding:
        image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        image_cd = (
            image_tensor_cd.unsqueeze(0).half().cuda()
            if image_tensor_cd is not None
            else None
        )
        cd_alpha = cd_alpha
        cd_beta = cd_beta
        print("image_cd", image_cd.shape)
        print(cd_alpha, cd_beta, args.noise_step)
        if model_name == "minigpt4":
            image_cd = image_cd.squeeze(0)

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt": qu},
                use_nucleus_sampling=args.sample,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                premature_layer=premature_layer,
                candidate_premature_layers=candidate_premature_layers,
                mature_layer=mature_layer,
                beam_search=beam_search,
                dola_decoding=dola_decoding,
                opera_decoding=opera_decoding,
                vcd_decoding=vcd_decoding,
                halc_decoding=halc_decoding,
                # HALC
                halc_assistant=halc_assistant_helper,
                # OPERA
                key_position=None,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                # VCD
                images_cd=image_cd,
                cd_alpha=cd_alpha,
                cd_beta=cd_beta,
            )

    output_text = out[0]

    print("decoder output text", output_text)

    img_save["caption"] = output_text

    if answer in output_text:
        img_save["answer"] = True
        hallucination_list.append(0)
    else:
        img_save["answer"] = False
        hallucination_list.append(1)

    # print("img_id: ", img_id)
    print("image_path: ", image_path)
    print("caption: ", output_text)

    # dump metric file
    generated_captions_path = os.path.join(
        base_dir,
        f"{model_name}_{decoding_strategy}_beams_{num_beams}_k_{k_candidate_num}_{dataset_name}_expand_ratio_{expand_ratio}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_samples}_hallucination_correction.json",
    )
    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")

hallucination_rate = sum(hallucination_list) / len(hallucination_list)
print("hallucination_rate", hallucination_rate)
