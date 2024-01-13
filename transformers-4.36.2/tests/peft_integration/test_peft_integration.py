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
import os
import tempfile
import unittest

from huggingface_hub import hf_hub_download

from transformers import AutoModelForCausalLM, OPTForCausalLM
from transformers.testing_utils import require_peft, require_torch, require_torch_gpu, slow, torch_device
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_peft
@require_torch
class PeftTesterMixin:
    peft_test_model_ids = ("peft-internal-testing/tiny-OPTForCausalLM-lora",)
    transformers_test_model_ids = ("hf-internal-testing/tiny-random-OPTForCausalLM",)
    transformers_test_model_classes = (AutoModelForCausalLM, OPTForCausalLM)


# TODO: run it with CI after PEFT release.
@slow
class PeftIntegrationTester(unittest.TestCase, PeftTesterMixin):
    """
    A testing suite that makes sure that the PeftModel class is correctly integrated into the transformers library.
    """

    def _check_lora_correctly_converted(self, model):
        """
        Utility method to check if the model has correctly adapters injected on it.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        is_peft_loaded = False

        for _, m in model.named_modules():
            if isinstance(m, BaseTunerLayer):
                is_peft_loaded = True
                break

        return is_peft_loaded

    def test_peft_from_pretrained(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`.
        This checks if we pass a remote folder that contains an adapter config and adapter weights, it
        should correctly load a model that has adapters injected on it.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                self.assertTrue(self._check_lora_correctly_converted(peft_model))
                self.assertTrue(peft_model._hf_peft_config_loaded)
                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    def test_peft_state_dict(self):
        """
        Simple test that checks if the returned state dict of `get_adapter_state_dict()` method contains
        the expected keys.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                state_dict = peft_model.get_adapter_state_dict()

                for key in state_dict.keys():
                    self.assertTrue("lora" in key)

    def test_peft_save_pretrained(self):
        """
        Test that checks various combinations of `save_pretrained` with a model that has adapters loaded
        on it. This checks if the saved model contains the expected files (adapter weights and adapter config).
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))

                    self.assertTrue("config.json" not in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

                    peft_model = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(peft_model))

                    peft_model.save_pretrained(tmpdirname, safe_serialization=False)
                    self.assertTrue("adapter_model.bin" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))

                    peft_model = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(peft_model))

    def test_peft_enable_disable_adapters(self):
        """
        A test that checks if `enable_adapters` and `disable_adapters` methods work as expected.
        """
        from peft import LoraConfig

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                peft_model.add_adapter(peft_config)

                peft_logits = peft_model(dummy_input).logits

                peft_model.disable_adapters()

                peft_logits_disabled = peft_model(dummy_input).logits

                peft_model.enable_adapters()

                peft_logits_enabled = peft_model(dummy_input).logits

                self.assertTrue(torch.allclose(peft_logits, peft_logits_enabled, atol=1e-12, rtol=1e-12))
                self.assertFalse(torch.allclose(peft_logits_enabled, peft_logits_disabled, atol=1e-12, rtol=1e-12))

    def test_peft_add_adapter(self):
        """
        Simple test that tests if `add_adapter` works as expected
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))
                # dummy generation
                _ = model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    def test_peft_add_adapter_from_pretrained(self):
        """
        Simple test that tests if `add_adapter` works as expected
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model_from_pretrained = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(model_from_pretrained))

    def test_peft_add_adapter_modules_to_save(self):
        """
        Simple test that tests if `add_adapter` works as expected when training with
        modules to save.
        """
        from peft import LoraConfig
        from peft.utils import ModulesToSaveWrapper

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

                model = transformers_class.from_pretrained(model_id).to(torch_device)
                peft_config = LoraConfig(init_lora_weights=False, modules_to_save=["lm_head"])
                model.add_adapter(peft_config)
                self._check_lora_correctly_converted(model)

                _has_modules_to_save_wrapper = False
                for name, module in model.named_modules():
                    if isinstance(module, ModulesToSaveWrapper):
                        _has_modules_to_save_wrapper = True
                        self.assertTrue(module.modules_to_save.default.weight.requires_grad)
                        self.assertTrue("lm_head" in name)
                        break

                self.assertTrue(_has_modules_to_save_wrapper)
                state_dict = model.get_adapter_state_dict()

                self.assertTrue("lm_head.weight" in state_dict.keys())

                logits = model(dummy_input).logits
                loss = logits.mean()
                loss.backward()

                for _, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue(param.grad is not None)

    def test_peft_add_adapter_training_gradient_checkpointing(self):
        """
        Simple test that tests if `add_adapter` works as expected when training with
        gradient checkpointing.
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))

                # When attaching adapters the input embeddings will stay frozen, this will
                # lead to the output embedding having requires_grad=False.
                dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)
                frozen_output = model.get_input_embeddings()(dummy_input)
                self.assertTrue(frozen_output.requires_grad is False)

                model.gradient_checkpointing_enable()

                # Since here we attached the hook, the input should have requires_grad to set
                # properly
                non_frozen_output = model.get_input_embeddings()(dummy_input)
                self.assertTrue(non_frozen_output.requires_grad is True)

                # To repro the Trainer issue
                dummy_input.requires_grad = False

                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        self.assertTrue(param.requires_grad)

                logits = model(dummy_input).logits
                loss = logits.mean()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue("lora" in name.lower())
                        self.assertTrue(param.grad is not None)

    def test_peft_add_multi_adapter(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`. This test tests if
        add_adapter works as expected in multi-adapter setting.
        """
        from peft import LoraConfig
        from peft.tuners.tuners_utils import BaseTunerLayer

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                is_peft_loaded = False
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                logits_original_model = model(dummy_input).logits

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                logits_adapter_1 = model(dummy_input)

                model.add_adapter(peft_config, adapter_name="adapter-2")

                logits_adapter_2 = model(dummy_input)

                for _, m in model.named_modules():
                    if isinstance(m, BaseTunerLayer):
                        is_peft_loaded = True
                        break

                self.assertTrue(is_peft_loaded)

                # dummy generation
                _ = model.generate(input_ids=dummy_input)

                model.set_adapter("default")
                self.assertTrue(model.active_adapters() == ["default"])
                self.assertTrue(model.active_adapter() == "default")

                model.set_adapter("adapter-2")
                self.assertTrue(model.active_adapters() == ["adapter-2"])
                self.assertTrue(model.active_adapter() == "adapter-2")

                # Logits comparison
                self.assertFalse(
                    torch.allclose(logits_adapter_1.logits, logits_adapter_2.logits, atol=1e-6, rtol=1e-6)
                )
                self.assertFalse(torch.allclose(logits_original_model, logits_adapter_2.logits, atol=1e-6, rtol=1e-6))

                model.set_adapter(["adapter-2", "default"])
                self.assertTrue(model.active_adapters() == ["adapter-2", "default"])
                self.assertTrue(model.active_adapter() == "adapter-2")

                logits_adapter_mixed = model(dummy_input)
                self.assertFalse(
                    torch.allclose(logits_adapter_1.logits, logits_adapter_mixed.logits, atol=1e-6, rtol=1e-6)
                )

                self.assertFalse(
                    torch.allclose(logits_adapter_2.logits, logits_adapter_mixed.logits, atol=1e-6, rtol=1e-6)
                )

                # multi active adapter saving not supported
                with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)

    @require_torch_gpu
    def test_peft_from_pretrained_kwargs(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained` + additional kwargs
        and see if the integraiton behaves as expected.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    @require_torch_gpu
    def test_peft_save_quantized(self):
        """
        Simple test that tests the basic usage of PEFT model save_pretrained with quantized base models
        """
        # 4bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear4bit")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

        # 8-bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

    @require_torch_gpu
    def test_peft_save_quantized_regression(self):
        """
        Simple test that tests the basic usage of PEFT model save_pretrained with quantized base models
        Regression test to make sure everything works as expected before the safetensors integration.
        """
        # 4bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear4bit")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname, safe_serialization=False)
                    self.assertTrue("adapter_model.bin" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

        # 8-bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname, safe_serialization=False)

                    self.assertTrue("adapter_model.bin" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

    def test_peft_pipeline(self):
        """
        Simple test that tests the basic usage of PEFT model + pipeline
        """
        from transformers import pipeline

        for model_id in self.peft_test_model_ids:
            pipe = pipeline("text-generation", model_id)
            _ = pipe("Hello")

    def test_peft_add_adapter_with_state_dict(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`. This test tests if
        add_adapter works as expected with a state_dict being passed.
        """
        from peft import LoraConfig

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                with self.assertRaises(ValueError):
                    model.load_adapter(peft_model_id=None)

                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")

                dummy_state_dict = torch.load(state_dict_path)

                model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=peft_config)
                with self.assertRaises(ValueError):
                    model.load_adapter(model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=None))
                self.assertTrue(self._check_lora_correctly_converted(model))

                # dummy generation
                _ = model.generate(input_ids=dummy_input)

    def test_peft_from_pretrained_hub_kwargs(self):
        """
        Tests different combinations of PEFT model + from_pretrained + hub kwargs
        """
        peft_model_id = "peft-internal-testing/tiny-opt-lora-revision"

        # This should not work
        with self.assertRaises(OSError):
            _ = AutoModelForCausalLM.from_pretrained(peft_model_id)

        adapter_kwargs = {"revision": "test"}

        # This should work
        model = AutoModelForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        model = OPTForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        adapter_kwargs = {"revision": "main", "subfolder": "test_subfolder"}

        model = AutoModelForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        model = OPTForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))
