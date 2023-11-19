import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import torch
from transformers import AutoTokenizer

import sys
sys.path.append("./MiniGPT-4/DoLa")
sys.path.append("./MiniGPT-4/DoLa/transformers-4.28.1")
sys.path.append("./MiniGPT-4/DoLa/transformers-4.28.1/src")
sys.path.append("./MiniGPT-4/DoLa/transformers-4.28.1/src/transformers")
sys.path.append("./MiniGPT-4/woodpecker")
sys.path.append('/data/xyq/bill/MiniGPT-4/woodpecker/mPLUG-Owl/mPLUG-Owl')

from woodpecker.vis_corrector import Corrector
from woodpecker.models.utils import extract_boxes, find_matching_boxes, annotate

from PIL import Image
from types import SimpleNamespace
import numpy as np
import cv2
import gradio as gr
import uuid
import random
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



# ========================================
#             Model Initialization
# ========================================
PROMPT_TEMPLATE = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: {question}
AI: '''

print('Initializing Chat')

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

# initialize corrector
args_dict = {
        'api_key': "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",
        'api_base': "https://api.openai.com/v1",
        'val_model_path': "Salesforce/blip2-flan-t5-xxl",
        'qa2c_model_path': "khhuang/zerofec-qa2claim-t5-base",
        'detector_config':"/data/xyq/bill/MiniGPT-4/woodpecker/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        'detector_model_path':"/data/xyq/bill/MiniGPT-4/woodpecker/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        'cache_dir': './cache_dir',
}

model_args = SimpleNamespace(**args_dict)
corrector = Corrector(model_args)

# initialize mplug-Owl
# pretrained_ckpt = "MAGAer13/mplug-owl-llama-7b"
# model = MplugOwlForConditionalGeneration.from_pretrained(
#     pretrained_ckpt,
#     torch_dtype=torch.bfloat16,
# ).to("cuda:1")
# image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
# processor = MplugOwlProcessor(image_processor, tokenizer)


chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

img_list = []
# img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/clock_on_a_beach.png"
img = '/data/xyq/bill/MiniGPT-4/beach_on_a_clock.png'
# img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/zoom_in_5.png"
# img = "/data/xyq/bill/MiniGPT-4/hallucinatory_image/inject.png"


@torch.no_grad()
def my_model_function(image_path, question, box_threshold, area_threshold):
    # create a temp dir to save uploaded imgs
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    unique_filename = str(uuid.uuid4()) + ".png"
    
    # temp_file_path = os.path.join(temp_dir, unique_filename)
    
    # success = cv2.imwrite(temp_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    temp_file_path = image_path

    # try:
    output_text, output_image = model_predict(temp_file_path, question, box_threshold, area_threshold) # 你的模型预测函数
    # finally:
    #     # remove temporary files.
    #     if os.path.exists(temp_file_path):
    #         os.remove(temp_file_path)
    
    return output_text, output_image


def get_minigpt_output(img_path, question):
    prompts = [PROMPT_TEMPLATE.format(question=question)]
    # image_list = [img_path]
    image_list = []

    print("input img_path: ", img_path)
    chat.upload_img(img_path, CONV_VISION, image_list)
    chat.encode_img(image_list, 38) 
    # chat.ask("describe the man in the image.", CONV_VISION)
    chat.ask(question, CONV_VISION)

    output_text, output_token, info = chat.answer(CONV_VISION, image_list)

    print("output_text: ", output_text)

    return output_text


def model_predict(image_path, question, box_threshold, area_threshold):
    
    temp_output_filepath = os.path.join("temp", str(uuid.uuid4()) + ".png")
    os.makedirs("temp", exist_ok=True)
    try:
        owl_output = get_minigpt_output(image_path, question)
        corrector_sample = {
            'img_path': image_path,
            'input_desc': owl_output,
            'query': question,
            'box_threshold': box_threshold,
            'area_threshold': area_threshold
        }
        corrector_sample = corrector.correct(corrector_sample)
        corrector_output = corrector_sample['output']
        
        extracted_boxes = extract_boxes(corrector_output)
        boxes, phrases = find_matching_boxes(extracted_boxes, corrector_sample['entity_info'])
        output_image = annotate(image_path, boxes, phrases)
        cv2.imwrite(temp_output_filepath, output_image)
        output_image_pil = Image.open(temp_output_filepath)
        output_text = f"mPLUG-Owl:\n{owl_output}\n\nCorrector:\n{corrector_output}"

        return output_text, output_image_pil
    finally:
        if os.path.exists(temp_output_filepath):
            os.remove(temp_output_filepath)

# raw_image = Image.open(img).convert('RGB')
            
# my_model_function(img, "What is the man holding in his hand?", 0.35, 0.02)
# my_model_function(img, "What is the man holding in his hand?", 0.05, 0.005)
my_model_function(img, "Describe this image with as many details as possible.", 0.05, 0.005)


# def create_multi_modal_demo():
#     with gr.Blocks() as instruct_demo:
#         with gr.Row():
#             with gr.Column():
#                 img = gr.Image(label='Upload Image')
#                 question = gr.Textbox(lines=2, label="Prompt")
                
#                 with gr.Accordion(label='Detector parameters', open=True):
#                     box_threshold = gr.Slider(minimum=0, maximum=1,
#                                      value=0.35, label="Box threshold")
#                     area_threshold = gr.Slider(minimum=0, maximum=1,
#                                      value=0.02, label="Area threshold")

#                 run_botton = gr.Button("Run")

#             with gr.Column():
#                 output_text = gr.Textbox(lines=10, label="Output")
#                 output_img = gr.Image(label="Output Image", type='pil')
                
#         inputs = [img, question, box_threshold, area_threshold]
#         outputs = [output_text, output_img]
        
#         examples = [
#             ["/data/xyq/bill/MiniGPT-4/woodpecker/examples/case1.jpg", "How many people in the image?"],
#             ["/data/xyq/bill/MiniGPT-4/woodpecker/examples/case2.jpg", "Is there any car in the image?"],
#             ["/data/xyq/bill/MiniGPT-4/woodpecker/examples/case3.jpg", "Describe this image."],
#         ]

#         gr.Examples(
#             examples=examples,
#             inputs=inputs,
#             outputs=outputs,
#             fn=my_model_function,
#             cache_examples=False,
#             run_on_click=True
#         )
#         run_botton.click(fn=my_model_function,
#                          inputs=inputs, outputs=outputs)
#     return instruct_demo

# description = """
# # Woodpecker: Hallucination Correction for MLLMs🔧
# **Note**: Due to network restrictions, it is recommended that the size of the uploaded image be less than **1M**.

# Please refer to our [github](https://github.com/BradyFU/Woodpecker) for more details.
# """

# with gr.Blocks(css="h1,p {text-align: center;}") as demo:
#     gr.Markdown(description)
#     with gr.TabItem("Multi-Modal Interaction"):
#         create_multi_modal_demo()

# demo.queue(api_open=False).launch(share=True)
