import argparse
import os, sys
import random
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa")
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa/transformers-4.28.1")
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa/transformers-4.28.1/src")
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa/transformers-4.28.1/src/transformers")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

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

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

img_list = []
# img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/clock_on_a_beach.png"
# img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/zoom_in_5.png"
img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/mask_irrelevant.png"


chat.upload_img(img, CONV_VISION, img_list)
chat.encode_img(img_list, 38) 
# chat.ask("describe the man in the image.", CONV_VISION)
chat.ask("What is the man holding in his hand?", CONV_VISION)

# useful token id
# "▁clock": 12006
# "▁sur": 1190
# "f": 29888

output_text, output_token, info = chat.answer(CONV_VISION, img_list)

print("output_text: ", output_text)

JSD_matrix = info["jsd_matrix"].permute(1,0)
# print("JSD_matrix", JSD_matrix)
JSD_matrix = JSD_matrix.flip(dims=[0]).cpu().numpy()*10000

output_tokens = info["output_tokens"]
decoded_tokens = info["decoded_tokens"]
clock_matrix = np.array(info["clock_matrix"]).reshape(-1,np.shape(JSD_matrix)[0] + 1).T
clock_matrix = np.flip(clock_matrix, axis=0)
surf_matrix = np.array(info["surf_matrix"]).reshape(-1,np.shape(JSD_matrix)[0] + 1).T
surf_matrix = np.flip(surf_matrix, axis=0)
all_layer_matrix = info["all_layer_matrix"]
# print("output_tokens: ", output_tokens)
print("decoded_tokens: ", decoded_tokens)
print("len of ", len(decoded_tokens))
column_labels = decoded_tokens
all_layer_matrix = np.array(all_layer_matrix)
print("shape of all: ", np.shape(all_layer_matrix))
all_layer_matrix_dim = np.shape(all_layer_matrix)
# all_layer_matrix.reshape(all_layer_matrix_dim[0], all_layer_matrix_dim[1], all_layer_matrix_dim[3])
all_layer_matrix = all_layer_matrix.reshape(all_layer_matrix_dim[0], all_layer_matrix_dim[1], all_layer_matrix_dim[3])


top_3 = np.argsort(all_layer_matrix, axis=2)[:,:,:3]
top_1 = np.argsort(all_layer_matrix, axis=2)[:,:,0]
# print("top_3: ", np.shape(top_3))
# print("top_1", top_1)

# sur_idx = output_tokens.tolist().index(1190)
# f_idx = output_tokens.tolist().index(29888)
# clock_idx = output_tokens.tolist().index(12006) #12006

# sur_matrix = top_3[sur_idx,:,:]
# f_matrix = top_3[f_idx,:,:]

# sur_token_matrix = []
# f_token_matrix = []
# for i, value in enumerate(sur_matrix):
#     line_word_vector = []
#     # print("value", value)
#     for j, j_value in enumerate(value):
#         # print("j_value", j_value)
#         line_word_vector.append(chat.model.llama_tokenizer.decode([j_value]))
#     sur_token_matrix.append(line_word_vector)

# for i, value in enumerate(f_matrix):
#     line_word_vector = []
#     # print("value", value)
#     for j, j_value in enumerate(value):
#         # print("j_value", j_value)
#         line_word_vector.append(chat.model.llama_tokenizer.decode([j_value]))
#     f_token_matrix.append(line_word_vector)

# print("f_token_matrix", f_token_matrix)
# print("sur_token_matrix", sur_token_matrix)


y_axis = np.arange(np.shape(JSD_matrix)[0])[::-1] * 2

full_y_axis = np.arange(np.shape(JSD_matrix)[0]+1)[::-1] * 2
#set figure size
plt.figure(figsize=(20, 10))
# Create the heatmap
ax = sns.heatmap(JSD_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=column_labels)

ax.set_yticklabels(y_axis)

# Add labels, title, etc.
plt.title('Jensen-Shannon Divergence')
plt.xlabel('output tokens')
plt.ylabel('premature layers')

# plt.savefig("figures/OH_JSD_matrix.png")


# Set figure size
plt.figure(figsize=(24, 10))

# Create a subplot for the first heatmap
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
ax1 = sns.heatmap(clock_matrix, annot=True, fmt=".2f", cmap="Purples", xticklabels=column_labels)
# ax1.set_yticklabels(y_axis)
ax1.set_yticklabels(full_y_axis)
ax1.set_title('Probability Matrix for "clock"')
ax1.set_xlabel('output tokens')
ax1.set_ylabel('Log likelihood')

# Create a subplot for the second heatmap
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
ax2 = sns.heatmap(surf_matrix, annot=True, fmt=".2f", cmap="Purples", xticklabels=column_labels)
# ax2.set_yticklabels(y_axis)
ax1.set_yticklabels(full_y_axis)
ax2.set_title('Probability Matrix for "surf"')
ax2.set_xlabel('output tokens')
ax2.set_ylabel('Log likelihood')

# Adjust layout so heatmaps do not overlap
plt.tight_layout()

# Save the figure
# plt.savefig("figures/probs_matrix.png")

plt.clf()

# # Insert the header at the beginning of the data (if not already present)
# header = ['top 1', 'top 2', 'top 3']
# f_token_matrix.insert(0, header)

# # Set figure size
# plt.figure(figsize=(8, 6))  # Adjust the size as needed

# # Create a table and remove the axes
# ax = plt.gca()
# ax.table(cellText=f_token_matrix, loc='center', cellLoc='center', colWidths=[0.3]*len(f_token_matrix[0]))
# ax.axis('off')

# # Optionally adjust the scale of the plot area for a large table
# ax.set_position([0, 0, 1, 1])

# # Save the figure
# plt.savefig("f_token_matrix.png", bbox_inches='tight')


# # # Save the figure
# plt.savefig("figures/hallucination_matrix.png")
