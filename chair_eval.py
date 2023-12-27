import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json

from context_density.halc import halc_assistant
from pycocotools.coco import COCO

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

parser.add_argument("--beam", type=int)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", "--num_samples", type=int, default=100)
parser.add_argument(
    "-v",
    "--verbosity",
    action="store_false",
    dest="verbosity",
    default=True,
    help="Verbosity. Default: True.",
)
parser.add_argument(
    "-b",
    "--beam-size",
    type=int,
    default=1,
    help="specify the beam size for halc.",
)
parser.add_argument(
    "-k",
    "--k-candidate-num",
    type=int,
    default=2,
    help="specify the k candidate number for halc.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./log/",
    help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
)
args = parser.parse_known_args()[0]


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
decoding_strategy = args.decoder
cfg = Config(args)
seed = args.seed
setup_seeds(cfg, seed)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
verbosity = args.verbosity
beam_size = args.beam_size
k_candidate_num = args.k_candidate_num
num_samples = args.num_samples
dataset_name = args.dataset_name
data_path = args.data_path
output_dir = args.output_dir

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("dataset_name: ", dataset_name)
    print("data_path: ", data_path)
    print("output_dir: ", output_dir)
    print("num_samples: ", num_samples)
    print("seed: ", seed)
    print(vis_processors["eval"].transform)




mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


annotation_file_path = args.data_path + 'annotations/instances_val2014.json'
# with open(args.data_path + '../annotations_trainval2014/annotations/instances_val2014.json', 'r') as f:
with open(annotation_file_path, 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

coco = COCO(annotation_file_path)


img_ids = coco.getImgIds()
# sample image ids
sampled_img_ids = random.sample(img_ids, num_samples)

print("sampled_img_ids", sampled_img_ids)

img_files = []
for cur_img_id in sampled_img_ids:
        
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

base_dir  = output_dir + args.model
if not os.path.exists(base_dir):
    os.mkdir(base_dir)


for img_id in tqdm(range(len(img_files))):
    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = args.data_path + img_file
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    
    # qu = "Please describe this image in detail."
    qu = "Generate a one sentence caption of the image."

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)


    halc_params = {"context_domain": "upper", "contrast_weight": 0.05, "context_window": 4, "expand_ratio": 0.15, "beam_size": args.beam_size, "k_candidate_num": args.k_candidate_num}
    halc_assistant_helper = halc_assistant(model, vis_processor=vis_processors, device=device, halc_params=halc_params)

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


    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu}, 
                use_nucleus_sampling=args.sample, 
                num_beams=args.beam,
                max_new_tokens=512,
                output_attentions=True,
                premature_layer=premature_layer,
                candidate_premature_layers=candidate_premature_layers,
                mature_layer=mature_layer,
                beam_search=True,
                dola_decoding=False,
                opera_decoding=True,
                halc_decoding=False,
                halc_assistant=halc_assistant_helper,
                key_position=None,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )
    img_save["caption"] = out[0]

    print("img_id: ", img_id)
    print("caption: ", out[0])
    input("done")

    # dump metric file
    with open(os.path.join(base_dir, 'ours-s_{}-t_{}-num_can_{}-p_{}.jsonl'.format(args.scale_factor, args.threshold, args.num_attn_candidates, args.penalty_weights)), "a") as f:
        json.dump(img_save, f)
        f.write('\n')
    


