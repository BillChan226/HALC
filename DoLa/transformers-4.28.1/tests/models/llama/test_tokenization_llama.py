# coding=utf-8
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

import os
import shutil
import tempfile
import unittest

from datasets import load_dataset

from transformers import (
    SPIECE_UNDERLINE,
    AddedToken,
    LlamaTokenizer,
    LlamaTokenizerFast,
    is_torch_available,
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    pass


@require_sentencepiece
@require_tokenizers
class LlamaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LlamaTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = LlamaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = LlamaTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    @unittest.skip("Let's wait for the fast tokenizer!")
    def test_save_pretrained(self):
        self.tokenizers_list += (self.rust_tokenizer_class, "hf-internal-testing/llama-tokenizer", {})
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files + the tokenizer.json file for the fast one
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))
                tokenizer_r_files = tuple(f for f in tokenizer_r_files if "tokenizer.json" not in f)
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=True
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=True)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=False
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=False)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it saved the tokenizer.json file
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

    @require_torch
    def test_batch_tokenization(self):
        if not self.test_seq2seq:
            return

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Longer text that will definitely require truncation.
                text = [
                    " UN Chief Says There Is No Military Solution in Syria",
                    " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for"
                    " Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons"
                    " will only worsen the violence and misery for millions of people.",
                ]
                try:
                    batch = tokenizer(
                        text=text,
                        max_length=3,
                        max_target_length=10,
                        return_tensors="pt",
                    )
                except NotImplementedError:
                    return
                self.assertEqual(batch.input_ids.shape[1], 3)
                # max_target_length will default to max_length if not specified
                batch = tokenizer(text, max_length=3, return_tensors="pt")
                self.assertEqual(batch.input_ids.shape[1], 3)

                batch_encoder_only = tokenizer(text=text, max_length=3, max_target_length=10, return_tensors="pt")
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    @unittest.skip("Unfortunately way too slow to build a BPE with SentencePiece.")
    def test_save_slow_from_fast_and_reload_fast(self):
        pass

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name,
                        additional_special_tokens=added_tokens,
                        **kwargs,  # , from_slow=True <- unfortunately too slow to convert
                    )
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    p_output = tokenizer_p.encode("Hey this is a <special> token")

                    cr_output = tokenizer_cr.encode("Hey this is a <special> token")

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[1, 4103, 689, 414, 313, 24784, 368, 2998, 408, 282, 3637, 25350, 29899, 9067, 414, 322, 282, 3637, 25350, 29899, 1457, 3018, 1312, 29899, 2151, 29897, 8128, 2498, 29899, 15503, 4220, 6956, 1973, 313, 13635, 29911, 29892, 402, 7982, 29899, 29906, 29892, 1528, 13635, 29911, 29874, 29892, 1060, 26369, 29892, 6652, 309, 29933, 814, 29892, 1060, 29931, 6779, 11410, 363, 18385, 17088, 7634, 11235, 313, 25103, 29965, 29897, 322, 18385, 17088, 28203, 313, 25103, 29954, 29897, 411, 975, 29871, 29941, 29906, 29974, 758, 3018, 1312, 4733, 297, 29871, 29896, 29900, 29900, 29974, 10276, 322, 6483, 1006, 3372, 3097, 1546, 435, 1165, 29892, 10772, 29911, 25350, 322, 323, 6073, 17907, 29889], [1, 350, 20161, 338, 8688, 304, 758, 29899, 14968, 6483, 21000, 8684, 284, 22540, 515, 443, 29880, 24025, 1426, 491, 14002, 368, 4195, 292, 373, 1716, 2175, 322, 1492, 3030, 297, 599, 15359, 29889], [1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="hf-internal-testing/llama-tokenizer",
            revision="0984d03108b1a041ed679bd253b6519b7e1a4778",
            padding=False,
        )


@require_torch
@require_sentencepiece
@require_tokenizers
class LlamaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "hf-internal-testing/llama-tokenizer"
        cls.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(checkpoint_name)
        cls.rust_tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint_name)
        return cls

    @require_torch
    def integration_tests(self):
        inputs = self.tokenizer(
            ["The following string should be properly encoded: Hello.", "But ird and ปี   ird   ด"],
            return_tensors="pt",
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                "input_ids": [
                    [1, 450, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889],
                    [1, 1205, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718],
                ],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            },
        )

    @slow
    def test_conversion(self):
        # This is excruciatingly slow since it has to recreate the entire merge
        # list from the original vocabulary in spm
        self.rust_tokenizer.save_pretrained("./out")
        with tempfile.TemporaryDirectory() as dirname:
            self.rust_tokenizer.save_pretrained(dirname)

            with open(os.path.join(dirname, "tokenizer.json"), "r") as f:
                old_serialized = f.read()

        new_tokenizer = convert_slow_tokenizer(self.tokenizer)
        with tempfile.NamedTemporaryFile() as f:
            new_tokenizer.save(f.name)
            # Re-opening since `f` is in bytes.
            new_serialized = open(f.name, "r").read()
            with open("out_tokenizer.json", "w") as g:
                g.write(new_serialized)

            self.assertEqual(old_serialized, new_serialized)

    def test_simple_encode_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(rust_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(pyth_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")
        self.assertEqual(rust_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(pyth_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])
        self.assertEqual(
            pyth_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )
        self.assertEqual(
            rust_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(pyth_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(pyth_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")
        self.assertEqual(rust_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(pyth_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(pyth_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")
        self.assertEqual(rust_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode(" "), [1, 259])
        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(pyth_tokenizer.encode("  "), [1, 1678])
        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [1, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

    def test_no_differences_showcase(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode(" "), [1, 259])
        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(pyth_tokenizer.encode("  "), [1, 1678])
        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [1, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    def test_no_differences_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.decode([869]), ".")
        self.assertEqual(rust_tokenizer.decode([869]), ".")

        self.assertEqual(pyth_tokenizer.decode([30112, 869]), "ا .")
        self.assertEqual(rust_tokenizer.decode([30112, 869]), "ا .")

    def test_no_differences_special_tokens(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    @unittest.skipIf(
        os.getenv("RUN_TOKENIZER_INTEGRATION", "0") == "0",
        "RUN_TOKENIZER_INTEGRATION=1 to run tokenizer integration tests",
    )
    def test_integration_test_xnli(self):
        import tqdm

        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        dataset = load_dataset("code_x_glue_ct_code_to_text", "go")
        for item in tqdm.tqdm(dataset["validation"]):
            string = item["code"]
            encoded1 = pyth_tokenizer.encode(string)
            encoded2 = rust_tokenizer.encode(string)

            self.assertEqual(encoded1, encoded2)

            decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("xnli", "all_languages")

        for item in tqdm.tqdm(dataset["train"]):
            for string in item["premise"].values():
                encoded1 = pyth_tokenizer.encode(string)
                encoded2 = rust_tokenizer.encode(string)

                self.assertEqual(encoded1, encoded2)

                decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
                decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

                self.assertEqual(decoded1, decoded2)
