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
import spacy

from transformers import AutoTokenizer

# initialize detector
args_dict = {
    "detector_config": "./woodpecker/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "detector_model_path": "./woodpecker/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    "cache_dir": "./cache_dir",
}


class halc_assistant:
    def __init__(self, model=None, vis_processor=None, device=None):
        model_args = SimpleNamespace(**args_dict)
        self.device = device
        self.detector = Detector(model_args)
        self.vis_processor = vis_processor
        self.model = model
        self.tagging = spacy.load("en_core_web_sm")
        self.tokenizer = self.model.llama_tokenizer
        token_vocab_dir = "./model_checkpoints/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/tokenizer.json"
        # token_vocab_dir = "/media/zhuokai/SN850X_4TB/contrast_decoding_LVLMs/model_checkpoints/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/tokenizer.json"
        if not os.path.exists(token_vocab_dir):
            temp = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",  # official model name for llama2-7b-chat-hf
            )
        with open(token_vocab_dir, "r") as f:
            self.token_vocab = json.load(f)

        self.token_vocab = self.token_vocab["model"]["vocab"]

        self.token_vocab = {value: key for key, value in self.token_vocab.items()}

    def update_img_path(self, img_path):
        print("img_path", img_path)
        self.detector_dict = {"img_path": img_path, "box_threshold": 0.1}

    def update_conv(self, conv):
        self.conv = conv


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
        # input_ids = input_ids
        # input_ids = input_ids.cpu().numpy().tolist()
        # input_ids = input_ids.numpy().tolist()
        # print("input_ids", input_ids)
        # input()
        # input_ids = torch.tensor(input_ids)
        output_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

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

    def draw_bbox(self, image, bbox, color="yellow", width=3):
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
            bbox[0] * im_width,
            bbox[1] * im_height,
            bbox[2] * im_width,
            bbox[3] * im_height,
        ]
        # Draw the rectangle on the image
        draw.rectangle(rect, outline=color, width=width)
        return image

    def context_density_embedding(self, entity, context_window=3):
        # context_window specifies the number of context windows
        entity = entity.strip(".")
        doc = self.tagging(entity)
        detect_info = {}
        
        if len(doc) < 1:
            detect_info["pos"] = "PUNC"
        else:
            detect_info["pos"] = doc[0].pos_

        print("entity", entity)
        print("pos", detect_info["pos"])

        valid_list = ["NOUN", "PROPN"]

        if detect_info["pos"] in valid_list:
            detect_info["status"] = "acctivated"
            self.detector_dict["named_entity"] = [entity]
            sample = self.detector.detect_objects(self.detector_dict)

            print("Detection: ", sample)

            # Assuming the first detected bounding box is the one related to the entity

            original_bbox = sample["entity_info"][entity]["bbox"]
            if len(original_bbox) == 0:
                target_bbox = [0.3, 0.3, 0.6, 0.6]
                detect_info["status"] = "bounding box not detected"
            else:
                area_list = [
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in original_bbox
                ]

                # get the index of the smallest bbox
                target_bbox_index = area_list.index(min(area_list))
                target_bbox = original_bbox[target_bbox_index]

            # target_bbox = original_bbox[0]

            # Calculate expanded bounding boxes for the given context window
            expanded_bboxes = [target_bbox]
            # for _ in range(1, context_window):
            #     # Each expansion is double the size of the previous level
            #     expanded_bboxes.append(self.expand_bbox(expanded_bboxes[-1], 1.5))

            expanded_bboxes.append(self.expand_bbox(expanded_bboxes[-1], -0.5))
            expanded_bboxes.append(self.expand_bbox(expanded_bboxes[-1], 5))

            # Load the original image
            image_path = sample["img_path"]
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
                save_path = f"./context_density/mnt/cropped_level_{i}.png"
                cropped_img.save(save_path)
                saved_paths.append(save_path)

            max_new_tokens = 300
            max_length = 2000
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
        
        else:
            detect_info["status"] = "invalid"
            embeds_list = None

        return embeds_list, detect_info


    def naive_focus_decoding(self, context_logits_list):
        # 
        # directly apply the detected box for decoding
        #
        contrast_logits = context_logits_list[0]
        return False, contrast_logits
    

    def context_curve_contrastive_decoding(self, context_logits_list):
        # 
        # this decoding method use the hallucination pattern for decoding
        #
        target_layer = context_logits_list[0]
        lower_layer = context_logits_list[1]
        upper_layer = context_logits_list[2]

        relative_top = 0.1

        # if relative_top > 0.0:
        #     final_logits = self.relative_top_filter(target_layer, relative_top)
        #     base_logits = upper_layer.log_softmax(dim=-1)
        #     mask = final_logits[0] < -1e3
        #     base_logits[0][mask] = -1e3
        # upper_contrast_logits = final_logits - base_logits

        # if relative_top > 0.0:
        #     final_logits = self.relative_top_filter(target_layer, relative_top)
        #     base_logits = lower_layer.log_softmax(dim=-1)
        #     mask = final_logits[0] < -1e3
        #     base_logits[0][mask] = -1e3
        # lower_contrast_logits = final_logits - base_logits

        upper_contrast_logits = target_layer - upper_layer
        lower_contrast_logits = target_layer - lower_layer

        # find out those tokens that are positive in both upper and lower contrast logits
        # Step 1: Identify positive logits
        positive_upper = upper_contrast_logits > 0
        positive_lower = lower_contrast_logits > 0

        # Step 2: Create a combined mask
        positive_both = np.logical_and(positive_upper.cpu().numpy(), positive_lower.cpu().numpy())

        contrast_logits = upper_layer.cpu().numpy() * positive_both

        contrast_logits = torch.tensor(contrast_logits).to(self.device)
 
        return False, contrast_logits
        
    def context_contrastive_decoding(self, context_logits_list, last_tokens):
        # 
        # this decoding method use the hallucination pattern as a filter for verification
        #
        hallucination_index = last_tokens[0]
        
        # print("ontext_logits_list[0]", context_logits_list[0])
        target_layer = context_logits_list[0]
        lower_layer = context_logits_list[1]
        upper_layer = context_logits_list[2]

        target_logits = target_layer[0][hallucination_index]
        upper_logits = upper_layer[0][hallucination_index]
        lower_logits = lower_layer[0][hallucination_index]
        upper_contrast_logits = target_logits - upper_logits
        lower_contrast_logits = target_logits - lower_logits

        if upper_contrast_logits > -2 and lower_contrast_logits > -2:
            skip_flag = True
        else:
            skip_flag = False

        # skip_flag = False

        return skip_flag, target_layer
    
    def your_decoding_method(self, context_logits_list):
        # 
        # put your decoding method here. refer to self.context_density_embedding for the defination of context_logits_list
        #
        pass

    # don't really know what relative_top_filter is for, but maybe this could help
    def relative_top_filter(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.FloatTensor:
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = filter_value
        return scores_normalized
