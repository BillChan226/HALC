
# Context-Density-Layers Contrastive Decoding for LVLMs (CDL)

## Installation

We use the MiniGPT-4 SOTA as the LVLM backbone for CDL. Please refer to the [Installation](#Installation) section in the [MiniGPT-4](#MiniGPT-4) README.

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

To reproduce our results, we use the **Llama-2-7b** as LLM backbone, which can be downloaded from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main). After downloading, modify the code accordingly [here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.

You also have to prepare the pretrained model checkpoints, which can be downloaded from [here](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing). After downloading, modify the code accordingly [here](eval_configs/minigpt4_llama2_eval.yaml#L10) at Line 10.







## CHAIR Evaluation of DoLa-based Contrastive Decoding LVLMs

The evaluation pipeline includes 2 steps.

### Running LVLM to generate captions and result file format-ready for CHAIR

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

### Evaluate with CHAIR

We use the generated `_chair.json` file, for example, `minigpt4_pretrain-llama2_coco_2000_chair.json` for the CHAIR evaluation. Under root directory, run

```
python eval_hallucination.py --metric chair --input_path [PATH_TO_.JSON_FILE] -v
```

The evaluation results will be printed in terminal.

## Getting Started with DoLa-based Contrastive Decoding LVLMs

### Test DoLa with their given example in the paper

run

```
python ./MiniGPT-4/toy_dola_eval.py --model-name ./models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8 --output-path output-path.json --num-gpus 1 --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32
```

Note: adding 32 in the early-exit-layers is crucial for reasonable output.

JSD for each candidate layer is printed and input at line 2720 of file ```./MiniGPT-4/DoLa/transformers-4.28.1/src/transformers/generation/utils.py```

### Test DoLa layer-wise contrast with miniGPT4-v for object hallucination

run a toy example:

```
python ./MiniGPT-4/contrast_decoding.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```

The image in ```hallucinatory_image/clock_on_a_beach.png``` is projected into the prefix of the language model as a context.



## License

This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with
BSD 3-Clause License [here](LICENSE_Lavis.md).
