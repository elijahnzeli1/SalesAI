import os
from huggingface_hub import HfApi, HfFolder, upload_folder

# Set these variables before running
MODEL_DIR = "./SalesA"  # Path to your model directory
REPO_ID = "Qybera/SalesA2"  # Change to your Hugging Face username/repo
MODEL_NAME = "SalesA AI"

# 1. Authenticate (run 'huggingface-cli login' in your terminal first)
# Or set HF_TOKEN as an environment variable
api = HfApi()

# 2. Create repo if it doesn't exist
if not api.repo_exists(REPO_ID, repo_type="model"):
    print(f"Creating new model repo: {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
else:
    print(f"Repo {REPO_ID} already exists.")

# 3. Upload all files from MODEL_DIR to the repo
print(f"Uploading files from {MODEL_DIR} to {REPO_ID} on Hugging Face Hub...")
upload_folder(
    repo_id=REPO_ID,
    folder_path=MODEL_DIR,
    repo_type="model",
    path_in_repo=".",
    commit_message=f"Add {MODEL_NAME} model files"
)
print("Upload complete.")

# 4. (Optional) Set model card metadata (tags, license, etc.)
# This is usually handled by the README.md YAML frontmatter, but you can also set it via the API if needed.

print(f"Model {MODEL_NAME} is now registered and uploaded to https://huggingface.co/{REPO_ID}")

# Instructions:
# 1. Make sure you have run 'huggingface-cli login' to authenticate.
# 2. Set MODEL_DIR and REPO_ID at the top of this script.
# 3. Run this script after your model and all files are exported.
# 4. Check your model page on the Hugging Face Hub! 