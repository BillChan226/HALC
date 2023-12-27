# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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

import numpy as np
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import SegformerImageProcessor


class SegformerImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_reduce_labels=False,
    ):
        size = size if size is not None else {"height": 30, "width": 30}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_reduce_labels = do_reduce_labels

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_reduce_labels": self.do_reduce_labels,
        }


def prepare_semantic_single_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

    image = Image.open(dataset[0]["file"])
    map = Image.open(dataset[1]["file"])

    return image, map


def prepare_semantic_batch_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

    image1 = Image.open(dataset[0]["file"])
    map1 = Image.open(dataset[1]["file"])
    image2 = Image.open(dataset[2]["file"])
    map2 = Image.open(dataset[3]["file"])

    return [image1, image2], [map1, map2]


@require_torch
@require_vision
class SegformerImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = SegformerImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = SegformerImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_reduce_labels"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 30, "width": 30})
        self.assertEqual(image_processor.do_reduce_labels, False)

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, reduce_labels=True)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})
        self.assertEqual(image_processor.do_reduce_labels, True)

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

    def test_call_segmentation_maps(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        maps = []
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
            maps.append(torch.zeros(image.shape[-2:]).long())

        # Test not batched input
        encoding = image_processing(image_inputs[0], maps[0], return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched
        encoding = image_processing(image_inputs, maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test not batched input (PIL images)
        image, segmentation_map = prepare_semantic_single_inputs()

        encoding = image_processing(image, segmentation_map, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched input (PIL images)
        images, segmentation_maps = prepare_semantic_batch_inputs()

        encoding = image_processing(images, segmentation_maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                2,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                2,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

    def test_reduce_labels(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)

        # ADE20k has 150 classes, and the background is included, so labels should be between 0 and 150
        image, map = prepare_semantic_single_inputs()
        encoding = image_processing(image, map, return_tensors="pt")
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 150)

        image_processing.do_reduce_labels = True
        encoding = image_processing(image, map, return_tensors="pt")
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)
