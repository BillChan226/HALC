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

from context_density.halc import halc_assistant

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
    def __init__(self, model, vis_processor, device="cuda:0", stopping_criteria=None, decoding_strategy = "greedy", hyper_params = None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        valid_decoding_strategies = ["greedy", "dola", "halc-dola", "halc-greedy", "halc-beam"]
        # assert decoding_strategy in valid_decoding_strategies:
        #     raise ValueError(f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}")
        assert decoding_strategy in valid_decoding_strategies, f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"
    
        self.decoding_strategy = decoding_strategy

        if self.decoding_strategy == "greedy":
            self.dola_decoding = False
            self.halc_decoding = False
            self.beam_search = False
        elif self.decoding_strategy == "dola":
            self.dola_decoding = True
            self.halc_decoding = False
            self.beam_search = False
        elif self.decoding_strategy == "halc-dola":
            self.dola_decoding = True
            self.halc_decoding = True
            self.beam_search = False
        elif self.decoding_strategy == "halc-greedy":
            self.dola_decoding = False
            self.halc_decoding = True
            self.beam_search = False
        elif self.decoding_strategy == "halc-beam":
            self.dola_decoding = True
            self.halc_decoding = True
            self.beam_search = True

        print(f"\033[42m####### Current Decoding Strategy: {self.decoding_strategy} #######\033[0m")

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )
        self.halc_params = hyper_params["halc_params"]
        self.halc_assistant = halc_assistant(self.model, vis_processor=self.vis_processor, device=self.device, halc_params=self.halc_params)

    def ask(self, text, conv):
        if (
            len(conv.messages) > 0
            and conv.messages[-1][0] == conv.roles[0]
            and conv.messages[-1][1][-6:] == "</Img>"
        ):  # last message is image.
            conv.messages[-1][1] = " ".join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

        self.halc_assistant.update_conv(conv)

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
        max_length=2000
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
            dola_decoding=self.dola_decoding,
            halc_decoding=self.halc_decoding,
            beam_search=self.beam_search,
            num_return_sequences=1,
            output_scores=True,
            premature_layer=premature_layer,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=mature_layer,
            return_dict_in_generate=False,
            halc_assistant=self.halc_assistant,
            beam_size=self.halc_params["beam_size"],
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        
        # print("conv", conv)
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token, info = self.model_generate(**generation_dict)
        # print("output_token: ", output_token)
        # print("info", info)
        # print("output_token", output_token)
        # sequences, scores = output_token.sequences, output_token.scores
        sequences = output_token
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
            self.halc_assistant.update_img_path(image)
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
