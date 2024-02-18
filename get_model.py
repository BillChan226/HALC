# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, snapshot_download

# huggingface_hub.hf_hub_download("lmsys/vicuna-7b-v1.1", local_dir="/root/autodl-fs/model_checkpoints")

# snapshot_download(repo_id="lmsys/vicuna-7b-v1.1", local_dir="/root/autodl-tmp/model_checkpoints")


repo_id = "meta-llama/Llama-2-7b-hf"
local_dir = "/home/czr/.cache/huggingface/hub/meta-llama/Llama-2-7b-hf"
cache_dir = local_dir + "/cache"
# while True:
if True:
    try:
        snapshot_download(cache_dir=cache_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False, # 不转为缓存乱码的形式, auto, Small files (<5MB) are duplicated in `local_dir` while a symlink is created for bigger files.
        resume_download=True,
        allow_patterns=["*.model", "*.json", "*.bin",
        "*.py", "*.md", "*.txt"],
        ignore_patterns=["*.safetensors", "*.msgpack",
        "*.h5", "*.ot", ],
        )
    # try:
    #     hf_hub_download(repo_id=repo_id, 
    #                     cache_dir=cache_dir,
    #                     resume_download=True,
    #                     local_dir=local_dir,
    #                     filename="model-00002-of-00002.safetensors")

    except Exception as e :
        print(e)