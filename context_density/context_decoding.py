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
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
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

img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/beach_on_a_clock.png"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/zoom_in_2.png"
# img = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/zoom_in_3.png"
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')


# early_exit_layers = np.arange(27,39)


early_exit_layer_idx = 38
print("early_exit_layer_idx: ", early_exit_layer_idx)

img_list = []

chat.upload_img(img, CONV_VISION, img_list)
chat.encode_img(img_list, early_exit_layer_idx) 


# chat.ask("describe the man in the image.", CONV_VISION)
chat.ask("What is the man holding in his handx?", CONV_VISION)

chat.code2_assistant.update_conv(CONV_VISION)

output_text, output_token, info = chat.answer(CONV_VISION, img_list)

print("output_text", output_text)
