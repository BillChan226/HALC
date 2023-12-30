import argparse
import os, sys
import random
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import StoppingCriteriaList

sys.path.append(".")

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub, SeparatorStyle, Conversation

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_llama2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("-d", type=str, default="dola", help="decoding strategy")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# img = '/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000043448.jpg'
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/beach_on_a_clock.png"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/pizza_with_topping.jpg"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/dog_tv.jpg"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/zoom_in_1.png"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/people_on_the_street.jpg"
# img = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000000196.jpg"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/zoom_in_2.png"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/zoom_in_3.png"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/dog_on_bed.jpg"
img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/breakfast.jpg"

# decoding_strategy = "halc-dola"
# decoding_strategy = "halc-greedy"
# decoding_strategy = "halc-beam"
decoding_strategy = args.d

halc_params = {"context_domain": "upper", "contrast_weight": 0.05, "context_window": 4, "expand_ratio": 0.15}

# halc_params = {"context_domain": "upper", "contrast_weight": 0.05, "context_window": 4, "expand_ratio": 0.2}
hyper_params = {"halc_params": halc_params}

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria, decoding_strategy=decoding_strategy, hyper_params=hyper_params)
print('Initialization Finished')


early_exit_layer_idx = 38
print("early_exit_layer_idx: ", early_exit_layer_idx)

img_list = []

chat.upload_img(img, CONV_VISION, img_list)
chat.encode_img(img_list, early_exit_layer_idx)


# chat.ask("Briefly describe the image.", CONV_VISION)
chat.ask("Please describe this image in detail.", CONV_VISION)
# chat.ask("What is the man holding in his hand?", CONV_VISION)
# chat.ask("Generate a one sentence caption of the image.", CONV_VISION)

output_text, output_token, info = chat.answer(CONV_VISION, img_list)

print("output_text", output_text)
print(f"\033[1;45m Final Decoded Text: {output_text} \033[0m")
