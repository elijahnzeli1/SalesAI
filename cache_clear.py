import shutil
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(cache_dir):
    print(f"Removing Hugging Face datasets cache at {cache_dir} ...")
    shutil.rmtree(cache_dir)
    print("Cache cleared.")
else:
    print("No Hugging Face datasets cache found.")