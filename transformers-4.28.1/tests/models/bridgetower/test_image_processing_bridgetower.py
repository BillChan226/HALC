# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
from typing import Dict, List, Optional, Union

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import BridgeTowerImageProcessor


class BridgeTowerImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        size_divisor: int = 32,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        do_center_crop: bool = True,
        image_mean: Optional[Union[float, List[float]]] = [0.48145466, 0.4578275, 0.40821073],
        image_std: Optional[Union[float, List[float]]] = [0.26862954, 0.26130258, 0.27577711],
        do_pad: bool = True,
        batch_size=7,
        min_resolution=30,
        max_resolution=400,
        num_channels=3,
    ):
        self.parent = parent
        self.do_resize = do_resize
        self.size = size if size is not None else {"shortest_edge": 288}
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_center_crop = do_center_crop
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "size_divisor": self.size_divisor,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to BridgeTowerImageProcessor,
        assuming do_resize is set to True with a scalar size and size_divisor.
        """
        if not batched:
            size = self.size["shortest_edge"]
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            scale = size / min(w, h)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size

            max_size = int((1333 / 800) * size)
            if max(newh, neww) > max_size:
                scale = max_size / max(newh, neww)
                newh = newh * scale
                neww = neww * scale

            newh, neww = int(newh + 0.5), int(neww + 0.5)
            expected_height, expected_width = (
                newh // self.size_divisor * self.size_divisor,
                neww // self.size_divisor * self.size_divisor,
            )

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width


@require_torch
@require_vision
class BridgeTowerImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = BridgeTowerImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = BridgeTowerImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "size_divisor"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)
        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)
        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)
        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_equivalence_pad_and_create_pixel_mask(self):
        # Initialize feature_extractors
        image_processing_1 = self.image_processing_class(**self.image_processor_dict)
        image_processing_2 = self.image_processing_class(do_resize=False, do_normalize=False, do_rescale=False)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad_and_return_pixel_mask" and calling the image processor return the same tensors
        encoded_images_with_method = image_processing_1.pad_and_create_pixel_mask(image_inputs, return_tensors="pt")
        encoded_images = image_processing_2(image_inputs, return_tensors="pt")

        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_mask"], encoded_images["pixel_mask"], atol=1e-4)
        )
