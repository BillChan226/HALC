import argparse
import os
import random
import shutil
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

# from decoder_zoo.Woodpecker.vis_corrector import Corrector
# from decoder_zoo.HaLC.context_density.halc import halc_assistant
# from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise

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
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(seed):
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
parser.add_argument("-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
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
parser.add_argument("--data_path", type=str, default="/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("-b", "--beam", type=int, default=3)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", "--num_samples", type=int, default=500)
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
    help="Post correction method such as Woodpecker, Lure.",
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
parser.add_argument(
    "--cd_beta",
    type=float,
    default=0.1,
    help="Beta param for VCD."
)
parser.add_argument(
    "--noise_step",
    type=int,
    default=3,
    help="Noise step for VCD."
)
parser.add_argument(
    "--generated_caption",
    type=str,
    help="Generated caption."
)

args = parser.parse_known_args()[0]

# print("args.gpu_id", args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

seed = args.seed
setup_seeds(seed)

device = torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
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
# generated_caption_path = args.generated_caption

loaded_json = []
loaded_keys = []
# with open(generated_caption_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         pair = json.loads(line)
#         loaded_json.append(pair)
#         loaded_keys.append(pair["image_id"])

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')




mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

annotation_file_path = args.data_path + 'annotations/instances_val2014.json'
caption_file_path = args.data_path + 'annotations/captions_val2014.json'
# with open(args.data_path + '../annotations_trainval2014/annotations/instances_val2014.json', 'r') as f:
with open(annotation_file_path, 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

coco = COCO(caption_file_path)

img_ids = coco.getImgIds()
# sample image ids
sampled_img_ids = random.sample(img_ids, num_samples)

refill_img_ids = []
for img in sampled_img_ids:
    print("img", img)
    if img not in loaded_keys:
        refill_img_ids.append(img)

print("refill_img_ids", refill_img_ids)
print("refill_img_ids", len(refill_img_ids))
input()


img_files = []
for cur_img_id in refill_img_ids:

    cur_img = coco.loadImgs(cur_img_id)[0]
    cur_img_path = cur_img["file_name"]
    img_files.append(cur_img_path)

img_dict = {}

categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )

# base_dir  = output_dir + args.model
# if not os.path.exists(base_dir):
#     os.makedirs(base_dir)


lost_image_folder = "/home/czr/HaLC/COCO_subset/"

lost_image_list = os.listdir(lost_image_folder)

print("lost_image_list", len(lost_image_list))
offlight = True

for img_id in tqdm(range(len(img_files))):

    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    # print("img_id", img_id)
    # if img_id != 321742 and offlight:
    #     continue
    # offlight = False

    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = args.data_path + img_file

    if image_path in lost_image_list:
        continue

    while True:
        # Path where the image will be copied if it fails to open
        lost_image_path = "/home/czr/HaLC/COCO_subset/" + img_file
        try:
            shutil.copy(image_path, lost_image_path)
            print("success")
            break
            
        except Exception as e:
            print("fail")
            continue

