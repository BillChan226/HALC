import argparse
import os, sys
import random
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt
import json
from context_density.detector import Detector
from types import SimpleNamespace
from PIL import Image, ImageDraw

# initialize detector
args_dict = {
    'detector_config':"./woodpecker/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    'detector_model_path':"./woodpecker/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    'cache_dir': './cache_dir',
}

class code2_assistant:
    def __init__(self, model=None, vis_processor=None, device=None):

        model_args = SimpleNamespace(**args_dict)
        self.device = device
        self.detector = Detector(model_args)
        self.vis_processor = vis_processor
        self.model = model
        self.tokenizer = self.model.llama_tokenizer
        token_vocab_dir = "/home/czr/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/tokenizer.json"
        with open(token_vocab_dir, "r") as f:
            self.token_vocab = json.load(f)

        self.token_vocab = self.token_vocab["model"]["vocab"]

        self.token_vocab = {value: key for key, value in self.token_vocab.items()}

    def update_img_path(self, img_path):
        print("img_path", img_path)
        self.detector_dict = {'img_path': img_path, 'box_threshold':0.1}

    def update_conv(self, conv):

        self.conv = conv

    # def check_word_complete(self, input_ids):
    #     input_ids = input_ids[0]
    #     input_ids = input_ids.cpu().numpy().tolist()

    #     # print("input_ids", input_ids)
    #     decoded_tokens = [
    #         self.tokenizer.decode([token_id]) for token_id in input_ids
    #     ]
        
    #     final_tokens = self.token_vocab[input_ids[-1]]
        

    #     output_text = self.tokenizer.decode(
    #         input_ids, skip_special_tokens=True
    #     )

    #     print("decoded_tokens", decoded_tokens)
    #     print("output_text", output_text)
    #     print("final_tokens: ", final_tokens)

    #     if "▁" in final_tokens:
    #         last_word_flag = True
    #         if len(decoded_tokens) < 2:
    #             last_word = output_text.split(" ")[-1]
    #         else:
    #             last_word = output_text.split(" ")[-2]
    #     else:
    #         last_word_flag = False
    #         last_word = "not completed yet!"
        
    #     return last_word_flag, last_word

    def check_word_complete(self, input_id):
        
        input_id = input_id.cpu().numpy().tolist()
        # print("input_id", input_id)
        final_tokens = self.token_vocab[input_id[0][0]]
        
        if "▁" in final_tokens or "." in final_tokens:
            last_word_flag = True
        else:
            last_word_flag = False
        
        return last_word_flag

    def get_last_word(self, input_ids):

        input_ids = input_ids[0]
        input_ids = input_ids.cpu().numpy().tolist()

        output_text = self.tokenizer.decode(
            input_ids, skip_special_tokens=True
        )

        last_word_flag = True
        last_word = output_text.split(" ")[-1]

        return last_word


    def expand_bbox(self, bbox, context_expansion_factor):
        """
        Expands the bounding box by a given expansion factor.
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        expanded_x_min = max(0, x_min - width * context_expansion_factor / 2)
        expanded_y_min = max(0, y_min - height * context_expansion_factor / 2)
        expanded_x_max = min(1, x_max + width * context_expansion_factor / 2)
        expanded_y_max = min(1, y_max + height * context_expansion_factor / 2)

        return [expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max]

    def draw_bbox(self, image, bbox, color='yellow', width=3):
        """
        Draws a bounding box on an image.

        :param image: The image on which to draw.
        :param bbox: The bounding box coordinates as a list of [x_min, y_min, x_max, y_max].
        :param color: The color of the box.
        :param width: The line width of the box.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        # Convert normalized bbox coordinates to absolute pixel values
        rect = [
            bbox[0] * im_width, bbox[1] * im_height,
            bbox[2] * im_width, bbox[3] * im_height
        ]
        # Draw the rectangle on the image
        draw.rectangle(rect, outline=color, width=width)
        return image

    def context_density_embedding(self, entity, context_window=3):
        # context_window specifies the number of context windows
        entity = entity.strip(".")
        self.detector_dict["named_entity"] = [entity]
        # self.detector_dict["named_entity"] = ["clock"]
        sample = self.detector.detect_objects(self.detector_dict)

        print("\nDetection: ", sample)

        # Assuming the first detected bounding box is the one related to the entity
        
        original_bbox = sample['entity_info'][entity]['bbox']
        area_list = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in sample['entity_info'][entity]['bbox']]

        # get the index of the smallest bbox
        target_bbox_index = area_list.index(min(area_list))
        target_bbox = sample['entity_info'][entity]['bbox'][target_bbox_index]
    
        # Calculate expanded bounding boxes for the given context window
        expanded_bboxes = [target_bbox]
        for _ in range(1, context_window):
            # Each expansion is double the size of the previous level
            expanded_bboxes.append(self.expand_bbox(expanded_bboxes[-1], 1.5))
    
        # Load the original image
        image_path = sample['img_path']
        original_image = Image.open(image_path).convert("RGB")

        # Crop images to the expanded bounding boxes
        cropped_images = []
        for bbox in expanded_bboxes:
            # Calculate the absolute coordinates of the bounding box
            im_width, im_height = original_image.size
            left = bbox[0] * im_width
            top = bbox[1] * im_height
            right = bbox[2] * im_width
            bottom = bbox[3] * im_height
            
            # Crop the image to the bounding box
            cropped_image = original_image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
        
        # Save the cropped images
        saved_paths = []
        for i, cropped_img in enumerate(cropped_images, start=1):
            save_path = f'./context_density/mnt/cropped_level_{i}.png'
            cropped_img.save(save_path)
            saved_paths.append(save_path)

        max_new_tokens=300
        max_length=2000
        embeds_list = []
        for i, cropped_img in enumerate(cropped_images, start=1):
            image = self.vis_processor(cropped_img).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image, 38)
            prompt = self.conv.get_prompt()
            # print("prompt: ", prompt)
            embs = self.model.get_context_emb(prompt, [image_emb])
            current_max_len = embs.shape[1] + max_new_tokens

            if current_max_len - max_length > 0:
                print(
                    "Warning: The number of tokens in current conversation exceeds the max length. "
                    "The model will not see the contexts outside the range."
                )

            begin_idx = max(0, current_max_len - max_length)
            embs = embs[:, begin_idx:]
            embeds_list.append(embs)
        # input()

        
        return embeds_list