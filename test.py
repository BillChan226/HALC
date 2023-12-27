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
import json, jsonlines

from decoder_zoo.Woodpecker.vis_corrector import Corrector
from decoder_zoo.HaLC.context_density.halc import halc_assistant

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict



# generated_captions_path = "/home/czr/HaLC/log/minigpt4/minigpt4_halc-beam_2_2_coco_0_2_generated_captions.json"

# loaded_json = []
# with open(generated_captions_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         loaded_json.append(json.loads(line))

# print(loaded_json)
generated_captions_path = "/home/czr/contrast_decoding_LVLMs/paper_exp/minigpt4_pretrain-llama2/halc-beam/coco/minigpt4_pretrain-llama2_halc-beam_2_2_coco_0_100_generated_captions.json"
# construct output file as input to CHAIR evaluation
# output format follows https://github.com/ruotianluo/self-critical.pytorch
formulated_output_dict = {}
# overall result
all_overall_scores = defaultdict(list)
# imgToEval per image result
img_to_eval_dict = {}

loaded_json = json.load(open(generated_captions_path))

annotation_file_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/annotations/captions_val2014.json"
# with open(args.data_path + '../annotations_trainval2014/annotations/instances_val2014.json', 'r') as f:
with open(annotation_file_path, 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

coco = COCO(annotation_file_path)



# to save memory, load 100 captions at a time
for start_idx in tqdm(
    range(0, len(loaded_json), 100), desc="Generating CHAIR Input"
):
    # define the current iteration end index
    end_idx = min(start_idx + 100, len(loaded_json))
    coco_res = coco.loadRes(
        loaded_json[start_idx:end_idx],
    )

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()
    coco_eval.evaluate()

    # keep track of the overall scores
    for metric, score in coco_eval.eval.items():
        all_overall_scores[metric].append(score)

    # imgToEval per image result
    for i, cur_img_id in enumerate(coco_res.getImgIds()):
        cur_eval_dict = coco_eval.evalImgs[i]
        # add caption to the eval dict
        cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
        img_to_eval_dict[cur_img_id] = cur_eval_dict

# overall result
overall_dict = {}
for metric, score in all_overall_scores.items():
    overall_dict[metric] = np.mean(score)
formulated_output_dict["overall"] = overall_dict
formulated_output_dict["imgToEval"] = img_to_eval_dict

# sanity check the results
if len(img_to_eval_dict) != num_samples:
    raise Exception(
        f"Resulting output_dict has number of images {len(img_to_eval_dict)} different from num_samples {num_samples}"
    )

if verbosity:
    print(
        f"\nGenerated {len(img_to_eval_dict)} samples results in CHAIR format."
    )

# save the formulated output dict
formulated_output_path = os.path.join(
    base_dir,
    f"{model_name}_{decoding_strategy}_{num_beams}_{k_candidate_num}_{dataset_name}_{seed}_{num_samples}_chair.json",
)

with open(formulated_output_path, "w") as f:
    json.dump(formulated_output_dict, f)
if verbosity:
    print(
        f"\nFormulated output matching CHAIR input format saved to {base_dir}."
    )

