import os
import json
import argparse
import sys
import random
import copy

sys.path.append("/home/czr/HaLC")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image


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
from decoder_zoo.HALC.context_density.halc import halc_assistant

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

import argparse

file_name = "/home/czr/HaLC/paper_result/llava-1.5/llava-1.5_halc-beam_beams_3_k_4_coco_expand_ratio_0.6_seed_2_max_tokens_64_samples_300_generated_captions.json"

loaded_json = []
with open(file_name, "r") as f:
    lines = f.readlines()
    for line in lines:
        loaded_json.append(json.loads(line))


# eliminate the items in loaded_json with the same key:
for i in range(len(loaded_json)):
    for j in range(i + 1, len(loaded_json)):
        if loaded_json[i]["image_id"] == loaded_json[j]["image_id"]:
            loaded_json.pop(j)
            break

# save loaded json

output_file_path = "/home/czr/HaLC/paper_result/llava-1.5/llava-1.5_halc-beam_beams_3_k_4_coco_expand_ratio_0.6_seed_2_max_tokens_64_samples_300_generated_captions_new.json"
with open(output_file_path, "a") as f:
    for i in loaded_json:
        json.dump(i, f)
        f.write("\n")
