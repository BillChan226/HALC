
# HaLC: Mitigating LVLMs Object Hallucination via Context-Density-Layer Contrastive Decoding

<div align="center">
  <img src="https://github.com/BillChan226/contrast_decoding_LVLMs/blob/main/.assets/hawk_logo.png" width="35%">
</div>

<!-- <a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2310.09478.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href='https://huggingface.co/spaces/Vision-CAIR/MiniGPT-v2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> <a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Gradio-Demo-blue'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=atFCwV2hSY4) -->


## :hammer_and_wrench: Installation

We use the MiniGPT-4 SOTA as the LVLM backbone for HALC. Please refer to the [Installation](#Installation) section in the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) README.

Briefly, run the following commands to install the required packages:

```
git clone https://github.com/BillChan226/contrast_decoding_LVLMs.git
cd contrast_decoding_LVLMs
conda env create -f environment.yml
conda activate minigptv

cd DoLa
pip install -e transformers-4.28.1
pip install datasets
```

We employ [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) as the external detector to bound hallucinatory objects. For quick installation:

```
pip install -U spacy
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm

cd woodpecker/GroundingDINO/
pip install -e .
```

Then download pre-trained model weights for DINO:
```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

If error `NameError: name '_C' is not defined` is reported, refer to [this issue](https://github.com/IDEA-Research/GroundingDINO/issues/8#issuecomment-1541892708) for a quick fix.

To reproduce our results, we use the **Llama-2-7b** as LLM backbone, which can be downloaded from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main). After downloading, modify the code accordingly [here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.

You also have to prepare the pretrained model checkpoints, which can be downloaded from [here](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing). After downloading, modify the code accordingly [here](eval_configs/minigpt4_llama2_eval.yaml#L10) at Line 10.


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



## :hourglass: Benchmarks
### :chair: CHAIR Evaluation of DoLa-based Contrastive Decoding LVLMs

The evaluation pipeline includes 2 steps.

#### Running LVLM to generate captions and result file format-ready for CHAIR

Following [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2305.10355.pdf), we used "Generate a short caption of the image" as the prompt to query LVLM for captions of the `2,000` images randomly sampled from [COCO 2014 Val](https://cocodataset.org/#download) datast. Under root directory, run

```
python generate_chair_input.py --num_samples 2000 --dataset_name coco --data_dir [COCO_DIR] --output_dir ./generated_captions/ -v
```

For a full list of command line input, run `python generate_chair_input.py -h`. Note that `[COCO_DIR]` is expected to contain both images and annotation files within the `annotations` subfolder. In other words, `[COCO_DIR]` should the the following structure:

```
COCO_DIR (val2024 for example)
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

#### Evaluate with CHAIR

We use the generated `_chair.json` file, for example, `minigpt4_pretrain-llama2_coco_2000_chair.json` for the CHAIR evaluation. Under root directory, run

```
python eval_hallucination.py --metric chair --chair_input_path [PATH_TO_.JSON_FILE] -v
```

The evaluation results will be printed in terminal.

## :wrench: Troubleshooting

Error installing `pattern`:

```
conda install -c conda-forge pattern
```

## :key: License

This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with
BSD 3-Clause License [here](LICENSE_Lavis.md).
