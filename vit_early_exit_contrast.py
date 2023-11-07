import argparse
import os, sys
import random
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa")
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa/transformers-4.28.1")
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa/transformers-4.28.1/src")
sys.path.append("/data/xyq/bill/MiniGPT-4/DoLa/transformers-4.28.1/src/transformers")
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



# img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/clock_on_a_beach.png"
img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/zoom_in_2.png"
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

inspected_clock_matrix = []
inspected_surf_matrix = []

early_exit_layers = np.arange(27,39)

column_labels = []
JSD_save_matrices = []
output_texts = []
for early_exit_layer_idx in early_exit_layers:

    print("early_exit_layer_idx: ", early_exit_layer_idx)
    img_list = []
    chat.upload_img(img, CONV_VISION, img_list)
    chat.encode_img(img_list, early_exit_layer_idx) 
    # chat.ask("describe the man in the image.", CONV_VISION)
    chat.ask("What is the man holding in his hand?", CONV_VISION)

    # output_text, output_token, info = chat.answer(CONV_VISION, img_list)
    output_text, output_token, info = chat.answer(CONV_VISION, img_list)

    CONV_VISION_LLama2 = Conversation(
        system="Give the following image: <Img>ImageContent</Img>. "
            "You will be able to see the image once I provide it to you. Please answer my questions.",
        roles=("<s>[INST] ", " [/INST] "),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    print("output_text: ", output_text)
    output_texts.append(".".join(output_text.split('.')[:2]))
    conv_dict = {'pretrain_llama2': CONV_VISION_LLama2}
    CONV_VISION = conv_dict[model_config.model_type]

    JSD_matrix = info["jsd_matrix"].permute(1,0)
    # print("JSD_matrix", JSD_matrix)
    JSD_matrix = JSD_matrix.flip(dims=[0]).cpu().numpy()*10000
    
    
    # print("JSD_matrix", np.shape(JSD_matrix))

    output_tokens = info["output_tokens"]
    decoded_tokens = info["decoded_tokens"]

    # find the index of "holding" in decoded_tokens
    if "holding" not in decoded_tokens:
        holding_idx = 4
    else:
        holding_idx = decoded_tokens.index("holding")
    
    column_labels.append(decoded_tokens[holding_idx-2:holding_idx+5])
    JSD_save_matrices.append(JSD_matrix[:,holding_idx-2:holding_idx+5])
    clock_matrix = np.array(info["clock_matrix"]).reshape(-1,np.shape(JSD_matrix)[0] + 1).T
    # print("clock_matrix", np.shape(info["clock_matrix"]))
    clock_matrix = np.flip(clock_matrix, axis=0)
    surf_matrix = np.array(info["surf_matrix"]).reshape(-1,np.shape(JSD_matrix)[0] + 1).T
    surf_matrix = np.flip(surf_matrix, axis=0)


    vit_column_labels = decoded_tokens[holding_idx:holding_idx+4]
    inspected_clock_matrix.append(clock_matrix[:,holding_idx:holding_idx+4].tolist())
    inspected_surf_matrix.append(surf_matrix[:,holding_idx:holding_idx+4].tolist())
    # print("inspected_clock_matrix", np.shape(inspected_clock_matrix))
    # input()

inspected_clock_matrix = np.flip(np.array(inspected_clock_matrix)[:,0,:], axis=0)
inspected_surf_matrix = np.flip(np.array(inspected_surf_matrix)[:,0,:], axis=0)

for jsd_matrix, column_label, vit_layer, output_text in zip(JSD_save_matrices, column_labels, early_exit_layers, output_texts):

    output_text.replace("\n", " ")
    llama_y_axis = np.arange(np.shape(JSD_matrix)[0])[::-1] * 2
    #set figure size
    plt.figure(figsize=(20, 10))
    # Create the heatmap
    ax = sns.heatmap(jsd_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=column_label)

    ax.set_yticklabels(llama_y_axis)

    # # Add labels, title, etc.
    plt.title('"JSD matrix w.r.t. LLM decoder layers')
    plt.xlabel('output tokens')
    plt.ylabel('early exit layers')
    # Adjust the subplot params to give space for the caption
    plt.subplots_adjust(bottom=0.2)

    # Add caption
    caption = output_text
    plt.figtext(0.5, 0.1, caption, wrap=True, horizontalalignment='center', fontsize=13)


    plt.savefig(f"figures/vit_early/vit_jsd_matrix{vit_layer}.png")

# Specify the filename
filename = 'output_texts.csv'

# Open the file in write mode
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Text', 'Early Exit Layer'])

    # Write the rows
    for text, layer in zip(output_texts, early_exit_layers):
        writer.writerow([text, layer])


y_axis = np.arange(np.shape(early_exit_layers)[0])[::-1]

y_axis = y_axis + 27
# Set figure size
plt.figure(figsize=(24, 10))

# Create a subplot for the first heatmap
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
ax1 = sns.heatmap(inspected_clock_matrix, annot=True, fmt=".2f", cmap="Greens")

ax1.set_yticklabels(y_axis)

# # Add labels, title, etc.
plt.title('"clock" prob w.r.t. ViT early exit layer ')
plt.xlabel('output tokens')
# ax1.set_yticks(y_axis)  # Set the y-tick positions
# ax1.set_yticklabels(output_texts) 

plt.ylabel('early exit layers')

# Create a subplot for the second heatmap
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
ax2 = sns.heatmap(inspected_surf_matrix, annot=True, fmt=".2f", cmap="Greens")
ax2.set_yticklabels(y_axis)
# ax1.set_yticks(y_axis)  # Set the y-tick positions
# ax1.set_yticklabels(output_texts)

# # Add labels, title, etc.
plt.title('"surfboard" prob w.r.t. ViT early exit layer ')
plt.xlabel('output tokens')
plt.ylabel('early exit layers')

# Adjust layout so heatmaps do not overlap
plt.tight_layout()

# Save the figure
plt.savefig("figures/vit_probs_matrix.png")

plt.clf()
