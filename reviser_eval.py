import argparse
import json
import os, sys
# delete path
sys.path.append("decoder_zoo/Woodpecker")
sys.path.append("decoder_zoo/LURE")
from vis_corrector import Corrector
from types import SimpleNamespace

from config import woodpecker_args_dict
import tqdm
import argparse
import os
import random
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from decoder_zoo.LURE.minigpt4.common.config import Config
from decoder_zoo.LURE.minigpt4.common.dist_utils import get_rank
from decoder_zoo.LURE.minigpt4.common.registry import registry
from decoder_zoo.LURE.minigpt4.conversation.conversation import Chat, CONV_VISION, Conversation, SeparatorStyle

from PIL import Image
from decoder_zoo.LURE.minigpt4.datasets.builders import *
from decoder_zoo.LURE.minigpt4.models import *
from decoder_zoo.LURE.minigpt4.processors import *
from decoder_zoo.LURE.minigpt4.runners import *
from decoder_zoo.LURE.minigpt4.tasks import *

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'Woodpecker: Hallucination Correction for MLLMs Hallucination Correction for MLLMs'.")
    parser.add_argument('--query', type=str, help="text query for MLLM", default="Please describe this image in detail.")
    parser.add_argument("--data_path", type=str, default="/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/", help="data path",)
    parser.add_argument("--output_dir", type=str, default="./log/", help="Output ditectory for saving test results. Default is './log/'.",)
    parser.add_argument("-c", "--caption-path", type=str, required=True, help="Path to the generated captions",)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization",)
    #### Woodpecker ####
    parser.add_argument('--cache-dir', type=str, help="dir for caching intermediate image", default='./cache_dir')
    parser.add_argument('--detector-config', type=str, help="Path to the detector config, in the form of 'path/to/GroundingDINO_SwinT_OGC.py' ")
    parser.add_argument('--detector-model', type=str, help="Path to the detector checkpoint, in the form of 'path/to/groundingdino_swint_ogc.pth' ")
    parser.add_argument('--api-key', type=str, help="API key for GPT service.")
    parser.add_argument('--api-base', type=str, help="API base link for GPT service.")
    
    #### LURE ####
    parser.add_argument("--cfg-path", default="decoder_zoo/LURE/eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("-m", "--max_new_tokens", type=int, default=64, help="max new tokens to generate for LURE")
    parser.add_argument('-r', '--reviser', type=str, help="which post-hoc corrector to use.")

    parser.add_argument("--continued_generation", type=str, default=None, help="path to continued caption generation.")

    args = parser.parse_args()
    reviser = args.reviser
    caption_path = args.caption_path
    seed = args.seed
    setup_seeds(seed)

    caption_data = []
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            caption_data.append(json.loads(line.strip()))


    valid_post_editing_strategies = ["lure", "woodpecker"]


    assert (
        reviser in valid_post_editing_strategies or reviser is None
    ), f"Invalid post correction strategy: {reviser}, should be in {valid_post_editing_strategies}"


    if reviser == "woodpecker":
        model_args = SimpleNamespace(**woodpecker_args_dict)
        corrector = Corrector(model_args)

    elif reviser == "lure":
        cfg = Config(args)
        max_new_tokens = args.max_new_tokens
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        print("vis_processor_cfg", vis_processor_cfg)
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

    qu = args.query
    data_path = args.data_path
    output_dir = args.output_dir
    input_captions_path = args.caption_path
    base_dir = output_dir + reviser

    continued_generation = args.continued_generation
    generated_caption = []
    generated_img_ids = []
    if continued_generation != None:
        
        with open(continued_generation, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                generated_caption.append(json.loads(line))
                generated_img_ids.append(json.loads(line)["image_id"])

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    corrected_caption_path = input_captions_path.replace("_dino", f"_{reviser}_dino").split("/")[-1]

    corrected_caption_path = os.path.join(
        base_dir,
        corrected_caption_path,
    )


    prefix = "COCO_val2014_"
    for idx, pair in tqdm(enumerate(caption_data), total=len(caption_data)):
        # if idx < 300:
        #     continue
        img_id = pair['image_id']
        caption = pair['caption']
        img_save = {}
        img_save["image_id"] = img_id

        if img_id in generated_img_ids:
            print("found existing img_id", img_id)
            img_save["caption"] = generated_caption[generated_img_ids.index(img_id)]["caption"]
            print("img_save", img_save)
            with open(corrected_caption_path, "a") as file:
                json.dump(img_save, file)
                file.write('\n')
            continue

        img_id = str(img_id).zfill(12)
        image_path = data_path + prefix + img_id + '.jpg'
        print("image_path", image_path)

        if reviser == "woodpecker":

            sample = {
            'img_path': image_path,
            'input_desc': caption,
            'query': qu,
            }
            
            corrected_sample = corrector.correct(sample)
            corrected_caption = corrected_sample['output']

        
        elif reviser == "lure":

            chat_state = Conversation(
                system='Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.', 
                roles=('Human', 'Assistant'), 
                messages=[['Human', '<Img><ImageHere></Img> ' + 'According to the picture, remove the information that does not exist in the following description: ' + caption]], 
                offset=2, 
                sep_style=SeparatorStyle.SINGLE, 
                sep='###', 
                sep2=None, 
                skip_next=False, 
                conv_id=None
            )

            image = Image.open(image_path).convert('RGB')
            img_list = []
            image = chat.vis_processor(image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))
            image_emb, _, = chat.model.encode_img(image)
            img_list.append(image_emb)
            corrected_caption = chat.answer(chat_state, img_list, max_new_tokens=max_new_tokens)

            # # float_list = [tensor.item() for tensor in plist]
            # result = {"image_id": image_id, "question": caption, "caption": output, "model": "LURE"}
            # print(result)
        print("original caption: ", caption)
        print("corrected caption: ", corrected_caption)


        img_save["caption"] = corrected_caption

        with open(corrected_caption_path, "a") as file:
            json.dump(img_save, file) #, ensure_ascii=False)
            file.write('\n')



