# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import (
        AutoProcessor,
        BertTokenizerFast,
        BlipImageProcessor,
        GPT2Tokenizer,
        InstructBlipProcessor,
        PreTrainedTokenizerFast,
    )


@require_vision
class InstructBlipProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = BlipImageProcessor()
        tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        qformer_tokenizer = BertTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-bert")

        processor = InstructBlipProcessor(image_processor, tokenizer, qformer_tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_qformer_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).qformer_tokenizer

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_additional_features(self):
        processor = InstructBlipProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
            qformer_tokenizer=self.get_qformer_tokenizer(),
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = InstructBlipProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, BlipImageProcessor)
        self.assertIsInstance(processor.qformer_tokenizer, BertTokenizerFast)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tokens = tokenizer(input_str, return_token_type_ids=False)
        encoded_tokens_qformer = qformer_tokenizer(input_str, return_token_type_ids=False)

        for key in encoded_tokens.keys():
            self.assertListEqual(encoded_tokens[key], encoded_processor[key])

        for key in encoded_tokens_qformer.keys():
            self.assertListEqual(encoded_tokens_qformer[key], encoded_processor["qformer_" + key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "attention_mask", "qformer_input_ids", "qformer_attention_mask", "pixel_values"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "attention_mask", "qformer_input_ids", "qformer_attention_mask", "pixel_values"],
        )
