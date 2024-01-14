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

from decoder_zoo.Woodpecker.vis_corrector import Corrector
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
    default="/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/",
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
parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta param for VCD.")
parser.add_argument("--noise_step", type=int, default=3, help="Noise step for VCD.")
parser.add_argument("--generated_caption", type=str, help="Generated caption.")
parser.add_argument(
    "--segment",
    type=str,
    help="Segment.",
    default=-1,
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
generated_caption_path = args.generated_caption
segment = int(args.segment)

loaded_json = []
loaded_keys = []
with open(generated_caption_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        pair = json.loads(line)
        loaded_json.append(pair)
        loaded_keys.append(pair["image_id"])

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

assert (
    decoding_strategy in valid_decoding_strategies
), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"

decoding_strategy = decoding_strategy
opera_decoding = False
dola_decoding = False
halc_decoding = False
vcd_decoding = False
beam_search = False

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
    model_args = SimpleNamespace(**args_dict)
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

annotation_file_path = args.data_path + "annotations/instances_val2014.json"
caption_file_path = args.data_path + "annotations/captions_val2014.json"
# with open(args.data_path + '../annotations_trainval2014/annotations/instances_val2014.json', 'r') as f:
with open(annotation_file_path, "r") as f:
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
# input()


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

base_dir = output_dir + args.model
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
    model, vis_processor=vis_processor, device=device, halc_params=halc_params
)

offlight = True

for idx, img_id in tqdm(enumerate(range(len(img_files)))):
    # if idx < segment:
    #     continue
    # if idx >= segment + 100:
    #     break

    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    print("img_id", img_id)
    # if img_id != 130076 and offlight:
    #     continue
    # offlight = False

    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = args.data_path + img_file
    # image_path = "/root/autodl-fs/HaLC/lost_images/" + img_file
    # print("image_path", image_path)

    try:
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)
    except Exception as e:
        #     # Print the error message
        print(f"Failed to open image {img_file}: {e}")
        continue

        # Path where the image will be copied if it fails to open

        # # Copy the image to the specified directory
        # shutil.copy(image_path, lost_image_path)

        # # Try to reopen the image from the new location
    #     try:
    #         raw_image = Image.open(lost_image_path).convert("RGB")
    #         image = vis_processors["eval"](raw_image).unsqueeze(0)
    #         image = image.to(device)
    #     except Exception as e:
    #         print(f"Failed to reopen image {img_file} from {lost_image_path}: {e}")
    #         # Handle the failure to reopen the image, if necessary
    # # print("image device", norm(image).device)

    qu = "Please describe this image in detail."
    # qu = "Generate a one sentence caption of the image."
    # qu = "Generate a short caption of the image."

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    # lm_early_exit_layers = [
    #     0,
    #     2,
    #     4,
    #     6,
    #     8,
    #     10,
    #     12,
    #     14,
    #     16,
    #     18,
    #     20,
    #     22,
    #     24,
    #     26,
    #     28,
    #     30,
    #     32,
    # ]
    lm_early_exit_layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        30,
    ]

    mature_layer = lm_early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = lm_early_exit_layers[:-1]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    halc_assistant_helper.update_input(img_path=image_path, input_prompt=qu)

    image_cd = None
    if decoding_strategy == "vcd":
        image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        image_cd = (
            image_tensor_cd.unsqueeze(0).half().cuda()
            if image_tensor_cd is not None
            else None
        )
        cd_alpha = cd_alpha
        cd_beta = cd_beta
        if model_name == "minigpt4":
            image_cd = image_cd.squeeze(0)

    try:
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
    except Exception as e:
        print(f"Failed to generate caption for image {img_file}: {e}")
        continue

    output_text = out[0]
    print("decoder output text", output_text)
    if post_correction == "woodpecker":
        sample = {
            "img_path": image_path,
            "input_desc": output_text,
            "query": "Generate a short caption of the image.",
        }

        corrected_sample = corrector.correct(sample)
        output_text = corrected_sample["output"]
        print("corrected output_text", output_text)
        input()

    img_save["caption"] = output_text

    # print("img_id: ", img_id)
    print("image_path: ", image_path)
    print("caption: ", output_text)

    # dump metric file
    # generated_captions_path = os.path.join(base_dir, f"{model_name}_{decoding_strategy}_beams_{num_beams}_k_{k_candidate_num}_{dataset_name}_expand_ratio_{expand_ratio}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_samples}_segment_{segment}_generated_captions_refill.json")
    generated_captions_path = os.path.join(
        base_dir,
        f"{model_name}_{decoding_strategy}_beams_{num_beams}_k_{k_candidate_num}_{dataset_name}_expand_ratio_{expand_ratio}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_samples}_generated_captions_refill.json",
    )

    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")
