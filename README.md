
# HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding

<div align="center">
  <img src="https://github.com/BillChan226/HALC/blob/main/.assets/hawk_logo.png" width="35%">
</div>


[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.17911-B21A1B)](https://arxiv.org/pdf/2311.17911.pdf)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://billchan226.github.io/projects/halc.html)
[![GitHub Stars](https://img.shields.io/github/stars/BillChan226/HALC?style=social)](https://github.com/BillChan226/HALC/stargazers)

This repository provides the official PyTorch implementation of the following paper: 
> [**HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding**]() <br>
> [Zhaorun Chen](https://billchan226.github.io/)<sup>1,\*</sup>, 
> [Zhuokai Zhao](https://zhuokai-zhao.com/)<sup>2,\*</sup>, 
> [Lingyu Gao](https://lygao.me/)<sup>3</sup>,
> [Hongyin Luo](https://luohongyin.github.io/) <sup>4</sup>,
> [Huaxiu Yao](https://www.huaxiuyao.io/) <sup>1</sup>, 
> [Jiawei Zhou](https://sites.harvard.edu/jzhou/)<sup>3</sup>
>
> <sup>1</sup>UNC-Chapel Hill, <sup>2</sup>University of Chicago, <sup>3</sup>Toyota Technological Institute at Chicago, <sup>4</sup>Massachusetts Institute of Technology <br>



## :hammer_and_wrench: Installation


To install, run the following commands to install the required packages:

```
git clone https://github.com/BillChan226/HALC.git
cd contrast_decoding_LVLMs
conda env create -f environment.yml
conda activate halc

python -m spacy download en_core_web_lg
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```

We employ [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) as the external detector to bound hallucinatory objects. To download pre-trained model weights for DINO:

```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```



## :bee: LVLM Backbones

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.
- Download [LLaMA-2 7B model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) and specify it at [Line 15](minigpt4/configs/models/minigpt4_llama2.yaml#L15)
- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at [Line 25](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25) of `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`.
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) of `minigpt4/configs/models/minigpt4_vicuna0.yaml`.
- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/minigpt4_eval.yaml#L8) of `eval_configs/minigpt4_eval.yaml`.


### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model`    | `llava-1.5` | Specify the MLLM model, this codebase supports `instructblip`, `minigpt4`, `llava-1.5`. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder, e.g., `COCO_2014/val2014/`. |
| `--pope-type`     | `random` | Type for POPE evaluation, supports `random`, `popular`, `adversarial`. |
| `--beam`   | `3` | Beam size for global search. Default: 1. |
| `--k-candidate-num`      | `4` | Number of generative focal fields for local search. Default: 2. |



## :hourglass: Benchmarks Evaluation
### :chair: CHAIR Evaluation of LVLMs Object Hallucination


#### Running LVLM to generate captions and result file format-ready for CHAIR

Following [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2305.10355.pdf), we used "Generate a short caption of the image" as the prompt to query LVLM for captions of the `2,000` images randomly sampled from [COCO 2014 Val](https://cocodataset.org/#download) datast. Under root directory, run

```
python generate_chair_input.py --num_samples 2000 --dataset_name coco --data_dir [COCO_DIR] --output_dir ./generated_captions/ --gpu-id 0 -v -d [Decoding Strategy]
```

For a full list of command line input, run `python generate_chair_input.py -h`. Note that `[COCO_DIR]` is expected to contain both images and annotation files within the `annotations` subfolder. In other words, `[COCO_DIR]` should the the following structure:

```
COCO_DIR (val2014 for example)
  - annotations
    - captions_val2014.json
    - captions_val2014.json
    - instances_train2014.json
    - instances_val2014.json
    - person_keypoints_train2014.json
    - person_keypoints_val2014.json
  - COCO_val2014_000000000042.jpg
  - COCO_val2014_000000000073.jpg
  ...
```

Upon completion, two files, `minigpt4_pretrain-llama2_coco_2000_generated_captions.json` and `minigpt4_pretrain-llama2_coco_2000_chair.json` should be generated under `generated_captions/minigpt4_pretrain-llama2/coco/` if `llama2` is the `model_type` used for `minigpt4`.


### :man_in_tuxedo: POPE Evaluation of LVLMs Object Hallucination

#### Running LVLM to generate captions and result file format-ready for POPE

Under root directory, run

```
python generate_pope_input.py --num_images 2000 --dataset_name coco --data_dir [COCO_DIR] --output_dir ./generated_captions/ --gpu-id 0 -v -d [Decoding Strategy]
```

### Evaluation

#### CHAIR Evaluation
We use the generated `_chair.json` file, for example, `minigpt4_pretrain-llama2_coco_2000_chair.json` for the CHAIR evaluation. Under root directory, run

```
python eval_hallucination.py --metric chair --chair_input_path [PATH_TO_.JSON_FILE] -v
```

#### POPE Evaluation

```
python eval_hallucination.py --metric pope --pope_answer_path [PATH_TO_MODEL_OUTPUT] --pope_question_path [PATH_TO_.POPE_QUESTION] -v
```

The evaluation results will be printed in terminal.



## :roller_coaster: Demo Playgrounds

### :eagle: HALC Demo
Run CDL demo on a [toy example](hallucinatory_image/beach_on_a_clock.png):
  
```
python context_density/context_decoding.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```

### ViT Early Exit Layers Demo 
Specify early_exit_layer_idx then run ViT early exit layers contrastive decoding:
  
```
python vit_early_exit_contrast.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```


### DoLA Demo

#### Test DoLa with their textual input

run

```
python toy_dola_eval.py --model-name ./models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8 --output-path output-path.json --num-gpus 1 --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32
```

Note: adding 32 in the early-exit-layers is crucial for reasonable output.

JSD for each candidate layer is printed and input at line 2720 of file ```DoLa/transformers-4.28.1/src/transformers/generation/utils.py```

#### Test DoLA with visual-textual input

run a toy example:

```
python contrast_decoding.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```

The [toy example](hallucinatory_image/beach_on_a_clock.png) is projected into the prefix of the language model as a context.




## :wrench: Troubleshooting

#### Error installing `GroundingDINO`: 

If error `NameError: name '_C' is not defined` is reported, refer to [this issue](https://github.com/IDEA-Research/GroundingDINO/issues/8#issuecomment-1541892708) for a quick fix.

#### Error installing `pattern`:

```
conda install -c conda-forge pattern
```

#### CUDA Error installing `GroundingDINO`:

```
conda install pytorch torchvision torchaudio pytorch-cuda=[YOUR NVIDIA CUDA VERSION] -c pytorch -c nvidia
```

#### runtimeError: Input type (float) and bias type (c10::Half) should be the same

simply reinstall torch==2.0.0 will most likely solve the issue

```
pip uninstall torch
pip install torch==2.0.0
```

## :key: License

This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with
BSD 3-Clause License [here](LICENSE_Lavis.md).
