# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_CAUSAL_LM_MAPPING, TextGenerationPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    require_accelerate,
    require_tf,
    require_torch,
    require_torch_gpu,
    require_torch_or_tf,
)

from .test_pipelines_common import ANY


@is_pipeline_test
@require_torch_or_tf
class TextGenerationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_CAUSAL_LM_MAPPING

    @require_torch
    def test_small_model_pt(self):
        text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="pt")
        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": (
                        "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope."
                        " oscope. FiliFili@@"
                    )
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"])
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": (
                            "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope."
                            " oscope. FiliFili@@"
                        )
                    }
                ],
                [
                    {
                        "generated_text": (
                            "This is a second test ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy"
                            " oscope. oscope. FiliFili@@"
                        )
                    }
                ],
            ],
        )

        outputs = text_generator("This is a test", do_sample=True, num_return_sequences=2, return_tensors=True)
        self.assertEqual(
            outputs,
            [
                {"generated_token_ids": ANY(list)},
                {"generated_token_ids": ANY(list)},
            ],
        )
        text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id
        text_generator.tokenizer.pad_token = "<pad>"
        outputs = text_generator(
            ["This is a test", "This is a second test"],
            do_sample=True,
            num_return_sequences=2,
            batch_size=2,
            return_tensors=True,
        )
        self.assertEqual(
            outputs,
            [
                [
                    {"generated_token_ids": ANY(list)},
                    {"generated_token_ids": ANY(list)},
                ],
                [
                    {"generated_token_ids": ANY(list)},
                    {"generated_token_ids": ANY(list)},
                ],
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="tf")

        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": (
                        "This is a test FeyFeyFey(Croatis.), s.), Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵"
                        " please,"
                    )
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"], do_sample=False)
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": (
                            "This is a test FeyFeyFey(Croatis.), s.), Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵"
                            " please,"
                        )
                    }
                ],
                [
                    {
                        "generated_text": (
                            "This is a second test Chieftain Chieftain prefecture prefecture prefecture Cannes Cannes"
                            " Cannes 閲閲Cannes Cannes Cannes 攵 please,"
                        )
                    }
                ],
            ],
        )

    def get_test_pipeline(self, model, tokenizer, processor):
        text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        return text_generator, ["This is a test", "Another test"]

    def test_stop_sequence_stopping_criteria(self):
        prompt = """Hello I believe in"""
        text_generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-gpt2")
        output = text_generator(prompt)
        self.assertEqual(
            output,
            [{"generated_text": "Hello I believe in fe fe fe fe fe fe fe fe fe fe fe fe"}],
        )

        output = text_generator(prompt, stop_sequence=" fe")
        self.assertEqual(output, [{"generated_text": "Hello I believe in fe"}])

    def run_pipeline_test(self, text_generator, _):
        model = text_generator.model
        tokenizer = text_generator.tokenizer

        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator("This is a test", return_full_text=False)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        outputs = text_generator("This is a test", return_full_text=True)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator(["This is great !", "Something else"], num_return_sequences=2, do_sample=True)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
            ],
        )

        if text_generator.tokenizer.pad_token is not None:
            outputs = text_generator(
                ["This is great !", "Something else"], num_return_sequences=2, batch_size=2, do_sample=True
            )
            self.assertEqual(
                outputs,
                [
                    [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                    [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                ],
            )

        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_full_text=True, return_text=True)
        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_full_text=True, return_tensors=True)
        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_text=True, return_tensors=True)

        # Empty prompt is slighly special
        # it requires BOS token to exist.
        # Special case for Pegasus which will always append EOS so will
        # work even without BOS.
        if (
            text_generator.tokenizer.bos_token_id is not None
            or "Pegasus" in tokenizer.__class__.__name__
            or "Git" in model.__class__.__name__
        ):
            outputs = text_generator("")
            self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        else:
            with self.assertRaises((ValueError, AssertionError)):
                outputs = text_generator("")

        if text_generator.framework == "tf":
            # TF generation does not support max_new_tokens, and it's impossible
            # to control long generation with only max_length without
            # fancy calculation, dismissing tests for now.
            return
        # We don't care about infinite range models.
        # They already work.
        # Skip this test for XGLM, since it uses sinusoidal positional embeddings which are resized on-the-fly.
        if tokenizer.model_max_length < 10000 and "XGLM" not in tokenizer.__class__.__name__:
            # Handling of large generations
            with self.assertRaises((RuntimeError, IndexError, ValueError, AssertionError)):
                text_generator("This is a test" * 500, max_new_tokens=20)

            outputs = text_generator("This is a test" * 500, handle_long_generation="hole", max_new_tokens=20)
            # Hole strategy cannot work
            with self.assertRaises(ValueError):
                text_generator(
                    "This is a test" * 500,
                    handle_long_generation="hole",
                    max_new_tokens=tokenizer.model_max_length + 10,
                )

    @require_torch
    @require_accelerate
    @require_torch_gpu
    def test_small_model_pt_bloom_accelerate(self):
        import torch

        # Classic `model_kwargs`
        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom",
            model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
        )
        self.assertEqual(pipe.model.device, torch.device(0))
        self.assertEqual(pipe.model.lm_head.weight.dtype, torch.bfloat16)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

        # Upgraded those two to real pipeline arguments (they just get sent for the model as they're unlikely to mean anything else.)
        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", device_map="auto", torch_dtype=torch.bfloat16)
        self.assertEqual(pipe.model.device, torch.device(0))
        self.assertEqual(pipe.model.lm_head.weight.dtype, torch.bfloat16)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

        # torch_dtype will be automatically set to float32 if not provided - check: https://github.com/huggingface/transformers/pull/20602
        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", device_map="auto")
        self.assertEqual(pipe.model.device, torch.device(0))
        self.assertEqual(pipe.model.lm_head.weight.dtype, torch.float32)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

    @require_torch
    @require_torch_gpu
    def test_small_model_fp16(self):
        import torch

        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", device=0, torch_dtype=torch.float16)
        pipe("This is a test")

    @require_torch
    @require_accelerate
    @require_torch_gpu
    def test_pipeline_accelerate_top_p(self):
        import torch

        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", device_map="auto", torch_dtype=torch.float16)
        pipe("This is a test", do_sample=True, top_p=0.5)
