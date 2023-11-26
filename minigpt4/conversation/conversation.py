import argparse
import time
from threading import Thread
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from context_density.utility import code2_assistant

from minigpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
        )

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop) :] == stop).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


class Chat:
    def __init__(self, model, vis_processor, device="cuda:0", stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )
        self.code2_assistant = code2_assistant(self.model, vis_processor=self.vis_processor, device=self.device)

    def ask(self, text, conv):
        if (
            len(conv.messages) > 0
            and conv.messages[-1][0] == conv.roles[0]
            and conv.messages[-1][1][-6:] == "</Img>"
        ):  # last message is image.
            conv.messages[-1][1] = " ".join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(
        self,
        conv,
        img_list,
        max_new_tokens=300,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.05,
        length_penalty=1,
        temperature=1.0,
        max_length=2000,
        dola_decoding=False,
        code2_decoding=False,
    ):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print("prompt: ", prompt)
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                "Warning: The number of tokens in current conversation exceeds the max length. "
                "The model will not see the contexts outside the range."
            )
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        lm_early_exit_layers = [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
        ]

        mature_layer = lm_early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = lm_early_exit_layers[:-1]
        premature_layer_dist = {l: 0 for l in candidate_premature_layers}

        # mode = "dola-static"
        # mature_layer = early_exit_layers[1]
        # premature_layer = None #early_exit_layers[0]
        # # candidate_premature_layers = None
        # candidate_premature_layers = early_exit_layers[:-1]
        # # if args.repetition_penalty is None:
        # #     args.repetition_penalty = 1.2

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
            dola_decoding=dola_decoding,
            code2_decoding=code2_decoding,
            num_return_sequences=1,
            output_scores=True,
            premature_layer=premature_layer,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=mature_layer,
            return_dict_in_generate=True,
            code2_kwargs=self.code2_assistant
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        
        # print("conv", conv)
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token, info = self.model_generate(**generation_dict)
        # print("output_token: ", output_token)
        # print("info", info)
        # print("output_token", output_token)
        sequences, scores = output_token.sequences, output_token.scores
        inputs_embeds = generation_dict["inputs_embeds"]
        # print("inputs_embeds.shape[-1]", inputs_embeds.shape[-1])
        # skip the tokens in the input prompt
        # gen_sequences = sequences[:, inputs_embeds.shape[-1]:][0, :]
        gen_sequences = sequences[:, 1:][0, :]

        gen_arr = gen_sequences.cpu().numpy()

        # print("gen_sequences", gen_sequences)

        info["output_tokens"] = gen_arr
        decoded_tokens = [
            self.model.llama_tokenizer.decode([token_id]) for token_id in gen_arr
        ]
        info["decoded_tokens"] = decoded_tokens

        output_token = output_token[0]
        output_text = self.model.llama_tokenizer.decode(
            gen_sequences, skip_special_tokens=True
        )
        # print("output_text", output_text)

        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy(), info

    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(
            self.model.llama_tokenizer, skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility

        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)

        return output

    def encode_img(self, img_list, early_exit_layer_idx=-1):
        image = img_list[0]
        self.image_path = image
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            self.code2_assistant.update_img_path(image)
            raw_image = Image.open(image).convert("RGB")
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image, early_exit_layer_idx)
        img_list.append(image_emb)

    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)
        msg = "Received."

        return msg
