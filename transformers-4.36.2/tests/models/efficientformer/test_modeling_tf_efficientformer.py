# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the TensorFlow EfficientFormer model. """

import inspect
import unittest
from typing import List

import numpy as np

from transformers import EfficientFormerConfig
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import cached_property, is_tf_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFEfficientFormerForImageClassification,
        TFEfficientFormerForImageClassificationWithTeacher,
        TFEfficientFormerModel,
    )
    from transformers.models.efficientformer.modeling_tf_efficientformer import (
        TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


if is_vision_available():
    from PIL import Image

    from transformers import EfficientFormerImageProcessor


class TFEfficientFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size: int = 13,
        image_size: int = 64,
        patch_size: int = 2,
        embed_dim: int = 3,
        num_channels: int = 3,
        is_training: bool = True,
        use_labels: bool = True,
        hidden_size: int = 128,
        hidden_sizes=[16, 32, 64, 128],
        num_hidden_layers: int = 7,
        num_attention_heads: int = 4,
        intermediate_size: int = 37,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        type_sequence_label_size: int = 10,
        initializer_range: float = 0.02,
        encoder_stride: int = 2,
        num_attention_outputs: int = 1,
        dim: int = 128,
        depths: List[int] = [2, 2, 2, 2],
        resolution: int = 2,
        mlp_expansion_ratio: int = 2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        self.num_attention_outputs = num_attention_outputs
        self.embed_dim = embed_dim
        self.seq_length = embed_dim + 1
        self.resolution = resolution
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.dim = dim
        self.mlp_expansion_ratio = mlp_expansion_ratio

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return EfficientFormerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            encoder_stride=self.encoder_stride,
            resolution=self.resolution,
            depths=self.depths,
            hidden_sizes=self.hidden_sizes,
            dim=self.dim,
            mlp_expansion_ratio=self.mlp_expansion_ratio,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TFEfficientFormerModel(config=config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = TFEfficientFormerForImageClassification(config)
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = TFEfficientFormerForImageClassification(config)

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFEfficientFormerModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_tf_common.py, as EfficientFormer does not use input_ids,
    inputs_embeds, attention_mask and seq_length.
    """

    all_model_classes = (
        (
            TFEfficientFormerModel,
            TFEfficientFormerForImageClassificationWithTeacher,
            TFEfficientFormerForImageClassification,
        )
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TFEfficientFormerModel,
            "image-classification": (
                TFEfficientFormerForImageClassification,
                TFEfficientFormerForImageClassificationWithTeacher,
            ),
        }
        if is_tf_available()
        else {}
    )

    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFEfficientFormerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=EfficientFormerConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="EfficientFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EfficientFormer does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[-1].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.asseretIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[-1].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "TFEfficientFormerForImageClassificationWithTeacher":
                del inputs_dict["labels"]

        return inputs_dict

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="EfficientFormer does not implement masked image modeling yet")
    def test_for_masked_image_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFEfficientFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)

        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_attention_outputs)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)

            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_attention_outputs)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    def test_compile_tf_model(self):
        # We use a simplified version of this test for EfficientFormer because it requires training=False
        # and Keras refuses to let us force that during functional construction
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Prepare our model
            model = model_class(config)
            # These are maximally general inputs for the model, with multiple None dimensions
            # Hopefully this will catch any conditionals that fail for flexible shapes
            functional_inputs = {
                key: tf.keras.Input(shape=val.shape[1:], dtype=val.dtype, name=key)
                for key, val in model.input_signature.items()
                if key in model.dummy_inputs
            }
            outputs_dict = model(functional_inputs)
            self.assertTrue(outputs_dict is not None)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class EfficientFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            EfficientFormerImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = TFEfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="tf")
        # forward pass
        outputs = model(**inputs, training=False)
        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = tf.constant([-0.0555, 0.4825, -0.0852])
        self.assertTrue(np.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_image_classification_head_with_teacher(self):
        model = TFEfficientFormerForImageClassificationWithTeacher.from_pretrained(
            "snap-research/efficientformer-l1-300"
        )
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="tf")
        # forward pass
        outputs = model(**inputs, training=False)
        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = tf.constant([-0.1312, 0.4353, -1.0499])
        self.assertTrue(np.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
