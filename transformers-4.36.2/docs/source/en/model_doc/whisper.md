<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Whisper

## Overview

The Whisper model was proposed in [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever.

The abstract from the paper is the following:

*We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.*

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ). The Tensorflow version of this model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://github.com/openai/whisper).

## Usage tips

- The model usually performs well without requiring any finetuning.
- The architecture follows a classic encoder-decoder architecture, which means that it relies on the [`~generation.GenerationMixin.generate`] function for inference.
- Inference is currently only implemented for short-form i.e. audio is pre-segmented into <=30s segments. Long-form (including timestamps) will be implemented in a future release.
- One can use [`WhisperProcessor`] to prepare audio for the model, and decode the predicted ID's back into text.

- To convert the model and the processor, we recommend using the following:

```bash
python src/transformers/models/whisper/convert_openai_to_hf.py --checkpoint_path "" --pytorch_dump_folder_path "Arthur/whisper-3" --convert_preprocessor True
```
The script will automatically determine all necessary parameters from the OpenAI checkpoint. A `tiktoken` library needs to be installed
to perform the conversion of the OpenAI tokenizer to the `tokenizers` version.

## Inference

Here is a step-by-step guide to transcribing an audio sample using a pre-trained Whisper model:

```python
>>> from datasets import load_dataset
>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration

>>> # Select an audio file and read it:
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> audio_sample = ds[0]["audio"]
>>> waveform = audio_sample["array"]
>>> sampling_rate = audio_sample["sampling_rate"]

>>> # Load the Whisper model in Hugging Face format:
>>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> # Use the model and processor to transcribe the audio:
>>> input_features = processor(
...     waveform, sampling_rate=sampling_rate, return_tensors="pt"
... ).input_features

>>> # Generate token ids
>>> predicted_ids = model.generate(input_features)

>>> # Decode token ids to text
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

>>> transcription[0]
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Whisper. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A fork with a script to [convert a Whisper model in Hugging Face format to OpenAI format](https://github.com/zuazo-forks/transformers/blob/convert_hf_to_openai/src/transformers/models/whisper/convert_hf_to_openai.py). 🌎
Usage example:
```bash
pip install -U openai-whisper
python convert_hf_to_openai.py \
    --checkpoint openai/whisper-tiny \
    --whisper_dump_path whisper-tiny-openai.pt
```

## WhisperConfig

[[autodoc]] WhisperConfig

## WhisperTokenizer

[[autodoc]] WhisperTokenizer
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_decode
    - decode

## WhisperTokenizerFast

[[autodoc]] WhisperTokenizerFast
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_decode
    - decode

## WhisperFeatureExtractor

[[autodoc]] WhisperFeatureExtractor
    - __call__

## WhisperProcessor

[[autodoc]] WhisperProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

<frameworkcontent>
<pt>

## WhisperModel

[[autodoc]] WhisperModel
    - forward
    - _mask_input_features

## WhisperForConditionalGeneration

[[autodoc]] WhisperForConditionalGeneration
    - forward
    - generate

## WhisperForCausalLM

[[autodoc]] WhisperForCausalLM
    - forward

## WhisperForAudioClassification

[[autodoc]] WhisperForAudioClassification
    - forward

</pt>
<tf>

## TFWhisperModel

[[autodoc]] TFWhisperModel
    - call

## TFWhisperForConditionalGeneration

[[autodoc]] TFWhisperForConditionalGeneration
    - call

</tf>
<jax>

## FlaxWhisperModel

[[autodoc]] FlaxWhisperModel
    - __call__

## FlaxWhisperForConditionalGeneration

[[autodoc]] FlaxWhisperForConditionalGeneration
    - __call__

## FlaxWhisperForAudioClassification

[[autodoc]] FlaxWhisperForAudioClassification
    - __call__

</jax>
</frameworkcontent>

