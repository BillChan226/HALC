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
from decoder_zoo.HaLC.context_density.detector import Detector
from types import SimpleNamespace
from PIL import Image, ImageDraw
import spacy
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
import random
from transformers import AutoTokenizer
from PIL import Image, ImageFilter


# initialize detector
args_dict = {
    "detector_config": "decoder_zoo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "detector_model_path": "decoder_zoo/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    "cache_dir": "decoder_zoo/HaLC/cache_dir",
}


class halc_assistant:
    def __init__(self, model=None, vis_processor=None, device=None, halc_params=None):
        model_args = SimpleNamespace(**args_dict)
        self.device = device
        self.detector = Detector(model_args)
        self.vis_processor = vis_processor
        self.model = model
        self.tagging = spacy.load("en_core_web_sm")
        self.tokenizer = self.model.llama_tokenizer
        self.halc_params = halc_params
        self.k_candidate_num = halc_params["k_candidate_num"]
        self.original_image = None
        self.model_backbone = halc_params["LVLM_backbone"]
        token_vocab_dir = "decoder_zoo/HaLC/context_density/tokenizer.json"
        if not os.path.exists(token_vocab_dir):
            temp = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",  # official model name for llama2-7b-chat-hf
            )
        with open(token_vocab_dir, "r") as f:
            self.token_vocab = json.load(f)

        self.token_vocab = self.token_vocab["model"]["vocab"]

        self.token_vocab = {value: key for key, value in self.token_vocab.items()}

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    def update_input(self, img_path, input_prompt):
        # print("img_path", img_path)
        self.detector_dict = {"img_path": img_path, "box_threshold": 0.1}
        self.prompt = input_prompt

    def check_word_complete(self, input_id):
        input_id = input_id.cpu().numpy().tolist()
        # print("input_id", input_id)
        final_tokens = self.token_vocab[input_id[0][0]]
        if "▁" in final_tokens or "." in final_tokens or input_id[0][0] == 2:
            last_word_flag = True
        else:
            last_word_flag = False

        return last_word_flag

    def get_last_word(self, input_ids):
        output_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        last_word_flag = True
        last_word = output_text.split(" ")[-1]

        return last_word

    def get_sequence_text(self, input_ids, skip_token_length=None):

        if skip_token_length == 0:
            output_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(input_ids[skip_token_length:], skip_special_tokens=True)

        return output_text

    def compute_bbox_size(self, bbox):
        """
        Computes the size of a bounding box.
        """
        # bbox = [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        return width * height

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

    def compute_jsd(self, p, q):
        # calculate the kl divergence
        def kl_divergence(p, q):
            return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)))

        # calculate the js divergence
        m = 0.5 * (p + q)
        jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        print(jsd)
        exit()

        return jsd

    def context_density_embedding(self, entity):
        # context_window specifies the number of context windows

        context_window = self.halc_params["context_window"]
        # expand_ratio = 0.1
        expand_ratio = self.halc_params["expand_ratio"]

        entity = entity.strip(".")
        doc = self.tagging(entity)
        detect_info = {}

        
        if len(doc) < 1:
            detect_info["pos"] = "PUNC"
        else:
            detect_info["pos"] = doc[0].pos_

        # print("entity", entity)
        # print("pos", detect_info["pos"])

        valid_list = ["NOUN", "PROPN"]

        if detect_info["pos"] in valid_list:
            detect_info["status"] = "acctivated"
            self.detector_dict["named_entity"] = [entity]
            sample = self.detector.detect_objects(self.detector_dict)

            # print("Detection: ", sample)
            # Assuming the first detected bounding box is the one related to the entity

            original_bbox = sample["entity_info"][entity]["bbox"]
            if len(original_bbox) > 2:
                detect_info["status"] = "invalid"
                embeds_list = None
                return embeds_list, detect_info

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
            if context_window == 1:
                # only the original one
                expanded_bboxes = [target_bbox]
            else:
                # check on the target box's size
                target_bbox_size = self.compute_bbox_size(target_bbox)
                # smallest size should be 0.2
                if target_bbox_size < 0.2:
                    # only increase the size
                    expanded_bboxes = [self.expand_bbox(target_bbox, -expand_ratio), target_bbox]
                    for _ in range(1, context_window-1):
                        # Each expansion is double the size of the previous level
                        expanded_bboxes.append(
                            self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                        )

                    # index of the original target box
                    self.target_bbox_index = 0
                else:
                    initial_ratio = np.sqrt(0.2 / target_bbox_size)
                    # expanded_bboxes = [self.expand_bbox(target_bbox, -initial_ratio)]
                    expanded_bboxes = [self.expand_bbox(target_bbox, -expand_ratio), target_bbox]
                    all_box_sizes = [self.compute_bbox_size(expanded_bboxes[0]), self.compute_bbox_size(expanded_bboxes[1])]
                    
                    for _ in range(1, context_window-1):
                        # Each expansion is double the size of the previous level
                        expanded_bboxes.append(
                            self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                        )
                        all_box_sizes.append(
                            self.compute_bbox_size(expanded_bboxes[-1])
                        )

                    # index of the original target box
                    self.target_bbox_index = min(
                        range(len(all_box_sizes)),
                        key=lambda i: abs(all_box_sizes[i] - target_bbox_size),
                    )

            # Load the original image
            image_path = sample["img_path"]
            self.original_image = Image.open(image_path)
            original_image = self.original_image.convert("RGB")

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

            # # Save the cropped images
            # saved_paths = []
            # for i, cropped_img in enumerate(cropped_images, start=1):
            #     save_path = f"./context_density/mnt/cropped_level_{i}.png"
            #     cropped_img.save(save_path)
            #     saved_paths.append(save_path)

            # get decoding for each context window

            embeds_list = []
            for i, cropped_img in enumerate(cropped_images, start=1):
                embs = self.get_model_embeds(cropped_img)
                embeds_list.append(embs)
        else:
            detect_info["status"] = "invalid"
            embeds_list = None

        return embeds_list, detect_info


    def get_model_embeds(self, image):

        if self.model_backbone == "minigpt4":
            max_new_tokens = 300
            max_length = 2000
        
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image, 38)
            
            prompt = self.prompt
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

        elif self.model_backbone == "llava-1.5":
            # image_emb = self.model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image)
            embs = self.vis_processor(image).unsqueeze(0).to(self.device)

        elif self.model_backbone == "blip":
            pass

        return embs

    def context_density_distortion_embedding(self, entity):
        # context_window specifies the number of context windows

        context_window = self.halc_params["context_window"]
        # expand_ratio = 0.1
        expand_ratio = self.halc_params["expand_ratio"]

        entity = entity.strip(".")
        doc = self.tagging(entity)
        detect_info = {}

        
        if len(doc) < 1:
            detect_info["pos"] = "PUNC"
        else:
            detect_info["pos"] = doc[0].pos_

        # print("entity", entity)
        # print("pos", detect_info["pos"])

        valid_list = ["NOUN", "PROPN"]

        if detect_info["pos"] in valid_list:
            detect_info["status"] = "acctivated"
            self.detector_dict["named_entity"] = [entity]
            sample = self.detector.detect_objects(self.detector_dict)

            # print("Detection: ", sample)
            # Assuming the first detected bounding box is the one related to the entity

            original_bbox = sample["entity_info"][entity]["bbox"]
            if len(original_bbox) > 2:
                detect_info["status"] = "invalid"
                embeds_list = None
                return embeds_list, detect_info

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
            if context_window == 1:
                # only the original one
                expanded_bboxes = [target_bbox]
            else:
                # check on the target box's size
                target_bbox_size = self.compute_bbox_size(target_bbox)
                # smallest size should be 0.2
                if target_bbox_size < 0.2:
                    # only increase the size
                    expanded_bboxes = [self.expand_bbox(target_bbox, -expand_ratio), target_bbox]
                    for _ in range(1, context_window-1):
                        # Each expansion is double the size of the previous level
                        expanded_bboxes.append(
                            self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                        )

                    # index of the original target box
                    self.target_bbox_index = 0
                else:
                    initial_ratio = np.sqrt(0.2 / target_bbox_size)
                    # expanded_bboxes = [self.expand_bbox(target_bbox, -initial_ratio)]
                    expanded_bboxes = [self.expand_bbox(target_bbox, -expand_ratio), target_bbox]
                    all_box_sizes = [self.compute_bbox_size(expanded_bboxes[0]), self.compute_bbox_size(expanded_bboxes[1])]
                    
                    for _ in range(1, context_window-1):
                        # Each expansion is double the size of the previous level
                        expanded_bboxes.append(
                            self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                        )
                        all_box_sizes.append(
                            self.compute_bbox_size(expanded_bboxes[-1])
                        )

                    # index of the original target box
                    self.target_bbox_index = min(
                        range(len(all_box_sizes)),
                        key=lambda i: abs(all_box_sizes[i] - target_bbox_size),
                    )

            # Load the original image
            image_path = sample["img_path"]
            self.original_image = Image.open(image_path)
            original_image = self.original_image.convert("RGB")


            im_width, im_height = original_image.size

            final_images = []  # List to store each modified image

            for bbox in expanded_bboxes:
                # original_image = self.original_image.convert("RGB")

                # original_image = self.draw_bbox(original_image, bbox, color="red", width=3)
                # Apply blur to the entire image
                blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=15))

                # Create a mask for the expanded bounding box
                left, top, right, bottom = [int(coord) for coord in (bbox[0] * im_width, bbox[1] * im_height, bbox[2] * im_width, bbox[3] * im_height)]
                mask = np.zeros((im_height, im_width), dtype=np.uint8)
                mask[top:bottom, left:right] = 255
                mask = Image.fromarray(mask)

                # Overlay the clear area over the blurred image
                final_image = Image.composite(original_image, blurred_image, mask)
                final_images.append(final_image)

            # # Save the distorted images
            saved_paths = []
            for i, cropped_img in enumerate(final_images, start=1):
                save_path = f"decoder_zoo/HaLC/mnt/cropped_level_{i}.png"
                cropped_img.save(save_path)
                saved_paths.append(save_path)
            # input("img saved!")

            # get decoding for each context window
            max_new_tokens = 300
            max_length = 2000
            embeds_list = []
            for i, cropped_img in enumerate(final_images, start=1):
                image = self.vis_processor(cropped_img).unsqueeze(0).to(self.device)
                image_emb, _ = self.model.encode_img(image, 38)
                prompt = self.prompt
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
        """
        directly apply the detected box for decoding
        """
        contrast_logits = context_logits_list[0]
        return False, contrast_logits

    def auto_regressive_decoding(self, context_logits_list):
        """
        directly apply the detected box for decoding
        """
        return True, None
    

    def context_curve_contrastive_decoding(self, context_logits_list):
        """
        this decoding method use the hallucination pattern for decoding
        """
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
        positive_both = np.logical_and(
            positive_upper.cpu().numpy(), positive_lower.cpu().numpy()
        )

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

        if upper_contrast_logits > -5 and lower_contrast_logits > -5:
            skip_flag = True
        else:
            skip_flag = False

        # skip_flag = False

        return skip_flag, target_layer

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

    def auto_contrastive_context_decoding(self, context_logits_list, last_tokens):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        hallucination_index = last_tokens[0]

        target_layer = context_logits_list[self.target_bbox_index]
        lower_layer = context_logits_list[0]
        upper_layer = context_logits_list[-1]

        target_logits = target_layer[0][hallucination_index]
        upper_logits = upper_layer[0][hallucination_index]
        lower_logits = lower_layer[0][hallucination_index]
        upper_contrast_logits = target_logits - upper_logits
        lower_contrast_logits = target_logits - lower_logits

        # if upper_contrast_logits > -2 and lower_contrast_logits > -2:
        if False:
            skip_flag = True
            return skip_flag, target_layer
        else:
            skip_flag = False

            non_target_layer_indices = [
                i
                for i in range(len(context_logits_list))
                if i != self.target_bbox_index
            ]
            # 1. Stacking all non-target context layer into a new dimension
            stacked_premature_layers = torch.stack(
                [context_logits_list[i] for i in non_target_layer_indices],
                dim=0,
            )

            # 2. Calculate the softmax values for mature_layer and all premature_layers
            softmax_mature_layer = F.softmax(
                context_logits_list[self.target_bbox_index], dim=-1
            )  # shape: (batch_size, num_features)
            softmax_premature_layers = F.softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 3. Calculate M, the average distribution
            M = 0.5 * (
                softmax_mature_layer[None, :, :] + softmax_premature_layers
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 4. Calculate log-softmax for the KL divergence
            log_softmax_mature_layer = F.log_softmax(
                context_logits_list[self.target_bbox_index], dim=-1
            )  # shape: (batch_size, num_features)
            log_softmax_premature_layers = F.log_softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 5. Calculate the KL divergences and then the JS divergences
            kl1 = F.kl_div(
                log_softmax_mature_layer[None, :, :], M, reduction="none"
            ).mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            kl2 = F.kl_div(log_softmax_premature_layers, M, reduction="none").mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

            # 6. Reduce the batchmean
            js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)

            premature_layer_index = non_target_layer_indices[
                int(js_divs.argmax().cpu().item())
            ]

            base_logits = context_logits_list[premature_layer_index]
            final_logits = context_logits_list[self.target_bbox_index]

            # final_logits = self.relative_top_filter(final_logits, relative_top=0.1)
            # base_logits = base_logits.log_softmax(dim=-1)
            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3
            contrast_logits = final_logits - base_logits

            # we always use the target layer for decoding
            return skip_flag, contrast_logits

    def contrastive_avg_context_decoding(self, context_logits_list, last_tokens):

        hallucination_index = last_tokens[0]

        target_layer = context_logits_list[self.target_bbox_index]
        lower_layer = context_logits_list[0]
        upper_layer = context_logits_list[-1]

        target_logits = target_layer[0][hallucination_index]
        upper_logits = upper_layer[0][hallucination_index]
        lower_logits = lower_layer[0][hallucination_index]
        upper_contrast_logits = target_logits - upper_logits
        lower_contrast_logits = target_logits - lower_logits

        if upper_contrast_logits > -2 and lower_contrast_logits > -2:
            skip_flag = True
            return skip_flag, target_layer
        else:
            skip_flag = False

            non_target_layer_indices = [
                i
                for i in range(len(context_logits_list))
                if i != self.target_bbox_index
            ]
            # stack all non-target context layer into a new dimension
            stacked_premature_layers = torch.stack(
                [context_logits_list[i] for i in non_target_layer_indices],
                dim=0,
            )

            # compute the average of premature layers
            premature_layer_avg = stacked_premature_layers.mean(dim=0)
            final_logits = context_logits_list[self.target_bbox_index]
            contrast_logits = final_logits - premature_layer_avg

            # we always use the target layer for decoding
            return skip_flag, contrast_logits

    def context_layer_contrastive_decoding(self, context_logits_list, last_tokens):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        skip_flag = False

        all_layer_indices = range(len(context_logits_list))

        # 1. Stacking all non-target context layer into a new dimension
        stacked_premature_layers = torch.stack(
            [context_logits_list[i] for i in all_layer_indices],
            dim=0,
        )

        # print("stacked_premature_layers", np.shape(stacked_premature_layers))
        # input()
        num_layers = len(stacked_premature_layers)
        jsd_matrix = torch.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(i+1, num_layers):
                M = 0.5 * (F.softmax(stacked_premature_layers[i], dim=-1) + F.softmax(stacked_premature_layers[j], dim=-1))
                kl1 = F.kl_div(F.log_softmax(stacked_premature_layers[i], dim=-1), M, reduction="batchmean")
                kl2 = F.kl_div(F.log_softmax(stacked_premature_layers[j], dim=-1), M, reduction="batchmean")
                jsd = 0.5 * (kl1 + kl2)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # Symmetric matrix

        # Find indices of max JSD
        # print("jsd_matrix.triu(diagonal=1)", jsd_matrix.triu(diagonal=1))
        max_jsd_flat_index = torch.argmax(jsd_matrix.triu(diagonal=1)) #.unbind()
        # layer_idx1, layer_idx2 = max_jsd_indices[0], max_jsd_indices[1]
        layer_idx1, layer_idx2 = np.unravel_index(max_jsd_flat_index.cpu().numpy(), jsd_matrix.shape)
        print("base_layer, final_layer: ", layer_idx1, layer_idx2)

        # # Update final_logits and base_logits
        # final_logits = context_logits_list[layer_idx1]
        # base_logits = context_logits_list[layer_idx2]

        context_domain = self.halc_params["context_domain"]
        # Update final_logits and base_logits
        if context_domain == "upper":
            base_logits = context_logits_list[layer_idx1]
            final_logits = context_logits_list[layer_idx2]
        elif context_domain == "lower":
            base_logits = context_logits_list[layer_idx2]
            final_logits = context_logits_list[layer_idx1]
        else:
            raise ValueError("Invalid context domain!")
        
        # final_logits = self.relative_top_filter(final_logits, relative_top=0.1)
        # base_logits = base_logits.log_softmax(dim=-1)
        mask = final_logits[0] < -1e3
        base_logits[0][mask] = -1e3

        contrast_weight = self.halc_params["contrast_weight"]

        # contrast_logits = final_logits - base_logits * 0.05
        contrast_logits = final_logits - base_logits * contrast_weight

        # we always use the target layer for decoding

        return skip_flag, contrast_logits


    def context_layer_multi_contrastive_decoding(self, context_logits_list, last_tokens):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        skip_flag = False
        k_candidate = self.k_candidate_num
        all_layer_indices = range(len(context_logits_list))


        stacked_premature_layers = torch.stack(
            [context_logits_list[i] for i in all_layer_indices],
            dim=0,
        )

        num_layers = len(stacked_premature_layers)
        jsd_matrix = torch.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(i+1, num_layers):
                M = 0.5 * (F.softmax(stacked_premature_layers[i], dim=-1) + F.softmax(stacked_premature_layers[j], dim=-1))
                kl1 = F.kl_div(F.log_softmax(stacked_premature_layers[i], dim=-1), M, reduction="batchmean")
                kl2 = F.kl_div(F.log_softmax(stacked_premature_layers[j], dim=-1), M, reduction="batchmean")
                jsd = 0.5 * (kl1 + kl2)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # Symmetric matrix


        # Find indices of top k_candidate JSD values
        upper_tri_flat = jsd_matrix.triu(diagonal=1).flatten()
        top_k_indices_flat = torch.topk(upper_tri_flat, k_candidate).indices
        rows = top_k_indices_flat // jsd_matrix.size(1)
        cols = top_k_indices_flat % jsd_matrix.size(1)
        top_k_indices = list(zip(rows.tolist(), cols.tolist()))

        contrast_logits_array = []
        context_domain = self.halc_params["context_domain"]
        contrast_weight = self.halc_params["contrast_weight"]

        for (layer_idx1, layer_idx2) in top_k_indices:
            # print("base_layer, final_layer: ", layer_idx1, layer_idx2)
            # (layer_idx1, layer_idx2) = top_k_indices[0]

            # Update final_logits and base_logits
            if context_domain == "upper":
                base_logits = context_logits_list[layer_idx1]
                final_logits = context_logits_list[layer_idx2]
            elif context_domain == "lower":
                base_logits = context_logits_list[layer_idx2]
                final_logits = context_logits_list[layer_idx1]
            else:
                raise ValueError("Invalid context domain!")

            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            # contrast_logits = final_logits - base_logits * 0.05
            contrast_logits = final_logits - base_logits * contrast_weight
            contrast_logits_array.append(contrast_logits)
            

        return skip_flag, contrast_logits_array


    def context_layer_double_multi_contrastive_decoding(self, context_logits_list, last_tokens):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        skip_flag = False
        if self.k_candidate_num % 2 != 0:
            raise ValueError("k_candidate_num must be even!")
        k_candidate = int(self.k_candidate_num/2)
        all_layer_indices = range(len(context_logits_list))


        stacked_premature_layers = torch.stack(
            [context_logits_list[i] for i in all_layer_indices],
            dim=0,
        )

        num_layers = len(stacked_premature_layers)
        jsd_matrix = torch.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(i+1, num_layers):
                M = 0.5 * (F.softmax(stacked_premature_layers[i], dim=-1) + F.softmax(stacked_premature_layers[j], dim=-1))
                kl1 = F.kl_div(F.log_softmax(stacked_premature_layers[i], dim=-1), M, reduction="batchmean")
                kl2 = F.kl_div(F.log_softmax(stacked_premature_layers[j], dim=-1), M, reduction="batchmean")
                jsd = 0.5 * (kl1 + kl2)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # Symmetric matrix


        # Find indices of top k_candidate JSD values
        upper_tri_flat = jsd_matrix.triu(diagonal=1).flatten()
        top_k_indices_flat = torch.topk(upper_tri_flat, k_candidate).indices
        rows = top_k_indices_flat // jsd_matrix.size(1)
        cols = top_k_indices_flat % jsd_matrix.size(1)
        top_k_indices = list(zip(rows.tolist(), cols.tolist()))

        contrast_logits_array = []
        context_domain = self.halc_params["context_domain"]
        contrast_weight = self.halc_params["contrast_weight"]

        for (layer_idx1, layer_idx2) in top_k_indices:

            base_logits = context_logits_list[layer_idx1]
            final_logits = context_logits_list[layer_idx2]

            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            contrast_logits = final_logits - base_logits * contrast_weight
            contrast_logits_array.append(contrast_logits)

            base_logits = context_logits_list[layer_idx2]
            final_logits = context_logits_list[layer_idx1]

            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            contrast_logits = final_logits - base_logits * contrast_weight
            contrast_logits_array.append(contrast_logits)
            
        return skip_flag, contrast_logits_array


    def clip_score_selection(self, candidate_intermediate_token_lists_array, beam_size, skip_token_length=0):
        

        if candidate_intermediate_token_lists_array == [None] * len(candidate_intermediate_token_lists_array) or self.original_image == None:
            # print("identical candidate lists: ", candidate_intermediate_token_lists_array)
            random.seed(8)
            selected_candidates = random.sample(range(len(candidate_intermediate_token_lists_array)), beam_size)

        else:
            candidate_texts = []
            for candidate_intermediate_token_lists in candidate_intermediate_token_lists_array:
                # print("candidate_intermediate_token_lists[0]", candidate_intermediate_token_lists[0])
                if self.model_backbone == "minigpt4":
                    skip_token_length = 0
                elif self.model_backbone == "llava-1.5":
                    skip_token_length = skip_token_length
                
                    # print("tokens_to_text", tokens_to_text)
                candidate_texts.append(self.get_sequence_text(candidate_intermediate_token_lists[0], skip_token_length))

            original_image = self.original_image
            
            # print("candidate_texts", candidate_texts)
            clip_inputs = self.clip_processor(text=candidate_texts, images=original_image, return_tensors="pt", padding=True, truncation=True)

            clip_outputs = self.clip_model(**clip_inputs)
            logits_per_image = clip_outputs.logits_per_image  # image-text similarity score
            clip_probs = logits_per_image.softmax(dim=1)[0]  # take the softmax to get the label probabilities

            # print("candidate lists:", candidate_intermediate_token_lists_array)
            # print("clip_probs:", clip_probs)
            
            # # get the top beam_size candidates
            clip_probs = clip_probs.cpu().numpy()
            # candidate_index = clip_probs.argsort()[-beam_size:][::-1]
            
            sorted_indices = clip_probs.argsort()[-len(candidate_intermediate_token_lists_array):][::-1]
            selected_texts = set()
            selected_candidates = []
            # print("sorted_indices:", sorted_indices)
            for idx in sorted_indices:

                # candidate_text = self.get_sequence_text(candidate_intermediate_token_lists_array[idx][0])
                candidate_text = candidate_texts[idx]
                # Check for uniqueness
                if candidate_text not in selected_texts:
                    selected_texts.add(candidate_text)
                    selected_candidates.append(idx)
                
                # Stop if enough candidates have been selected
                if len(selected_candidates) == beam_size:
                    break

        # print("selected_candidates:", selected_candidates)
        if len(selected_candidates) < beam_size:
            # copy the first one
            for _ in range(beam_size - len(selected_candidates)):
                selected_candidates.append(selected_candidates[0])

        candidate_index = selected_candidates
        # print("candidate_index:", candidate_index)
        # input()

        return candidate_index

    def random_selection(self, candidate_intermediate_token_lists_array, beam_size):

        random.seed(8)
        candidate_index = random.sample(range(len(candidate_intermediate_token_lists_array)), beam_size)
            
        return candidate_index



