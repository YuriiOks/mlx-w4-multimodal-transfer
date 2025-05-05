#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PROJECT_NAME="Multimodal Transfer Learning / Image Captioning"
TEAM_NAME="Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)"
COPYRIGHT_YEAR="2025"
CREATION_DATE="2025-05-05" # Set to May 5th
VENV_NAME=".venv"
PYTHON_CMD="python3" # Use python3, change if needed

echo "🚀 Starting Project Setup: $PROJECT_NAME by $TEAM_NAME"
echo "-----------------------------------------------------"

# --- Helper Function to Create Python Files with Headers ---
create_py_file() {
  local filepath="$1"
  local description="$2"
  # Ensure directory exists
  mkdir -p "$(dirname "$filepath")"
  # Create file with header
  cat << EOF > "$filepath"
# $PROJECT_NAME
# File: $filepath
# Copyright (c) $COPYRIGHT_YEAR $TEAM_NAME
# Description: $description
# Created: $CREATION_DATE
# Updated: $CREATION_DATE

# Basic pass statement or main block for runnable scripts
if __name__ == "__main__":
    pass
EOF
  echo "  📄 Created: $filepath"
}

# --- 1. Create Directory Structure ---
echo "📁 Creating Directory Structure..."
mkdir -p app
mkdir -p data/cache data/features/clip-vit-base-patch32 # Example feature dir
mkdir -p docs/presentation/images docs/notes
mkdir -p logs
mkdir -p models/image_captioning # Specific task subdir
mkdir -p notebooks
mkdir -p scripts
mkdir -p tests/common tests/data tests/models tests/training
mkdir -p src/common src/data src/models/pytorch src/models/mlx src/training/pytorch src/training/mlx src/inference src/deployment
mkdir -p utils
echo "  ✅ Directory structure created."
echo "-----------------------------------------------------"

# --- 2. Create Placeholder __init__.py Files ---
echo "🐍 Creating __init__.py files..."
find app src tests utils -type d -exec touch {}/__init__.py \;
# Clean up potential root-level __init__.py if find creates it
[ -f ./__init__.py ] && rm ./__init__.py
echo "  ✅ __init__.py files created."
echo "-----------------------------------------------------"

# --- 3. Create Core Python Files with Headers ---
echo "📄 Creating Core Python Files..."

# App files
create_py_file "app/app.py" "Main Streamlit application script."
create_py_file "app/sidebar.py" "Sidebar UI layout and controls."
create_py_file "app/input_section.py" "Input methods (upload, generate)."
create_py_file "app/prediction_section.py" "Prediction display logic."
create_py_file "app/model_loader.py" "Handles loading PT/MLX models + config from checkpoint."
create_py_file "app/preprocessing.py" "Image preprocessing for app input."
create_py_file "app/utils_streamlit.py" "Streamlit-specific helpers."

# Src/Common
create_py_file "src/common/constants.py" "Shared constants (PAD_ID, START_ID, etc.)."
create_py_file "src/common/tokenizer.py" "Text tokenizer setup (e.g., wrapping Hugging Face Tokenizers)."
create_py_file "src/common/metrics.py" "Evaluation metrics calculation (BLEU, ROUGE, etc.)."

# Src/Data
create_py_file "src/data/image_datasets.py" "PyTorch Dataset classes for image-caption data (e.g., Flickr30k)."
create_py_file "src/data/feature_datasets.py" "(Optional) Dataset classes loading pre-computed image features."
create_py_file "src/data/image_transforms.py" "Image preprocessing and augmentation functions/classes."
create_py_file "src/data/dataloader.py" "Functions to create PyTorch DataLoaders."

# Src/Models
create_py_file "src/models/encoder_wrapper.py" "Handles loading ViT/CLIP and extracting features."
create_py_file "src/models/pytorch/modules.py" "PyTorch Transformer Decoder building blocks."
create_py_file "src/models/pytorch/caption_model.py" "PyTorch Encoder-Decoder model assembly."
create_py_file "src/models/mlx/modules.py" "MLX Transformer Decoder building blocks."
create_py_file "src/models/mlx/caption_model.py" "MLX Encoder-Decoder model assembly."

# Src/Training
create_py_file "src/training/pytorch/trainer.py" "PyTorch training/evaluation loop logic."
create_py_file "src/training/pytorch/optim.py" "(Optional) PyTorch optimizer/scheduler setup helpers."
create_py_file "src/training/pytorch/checkpoint.py" "PyTorch save/load checkpoint functions (incl. config)."
create_py_file "src/training/mlx/trainer.py" "MLX training/evaluation loop logic."
create_py_file "src/training/mlx/optim.py" "(Optional) MLX optimizer/scheduler setup helpers."
create_py_file "src/training/mlx/checkpoint.py" "MLX save/load checkpoint functions (incl. config)."

# Src/Inference
create_py_file "src/inference/generator_pt.py" "PyTorch caption generation (greedy, beam search)."
create_py_file "src/inference/generator_mlx.py" "MLX caption generation (greedy, beam search)."

# Src/Deployment (Optional)
create_py_file "src/deployment/config_loader_app.py" "Specific config loading for deployment environment."

# Utils
create_py_file "utils/config.py" "Configuration loading functions/classes."
create_py_file "utils/device_setup.py" "PyTorch device selection (CPU/MPS/CUDA)."
create_py_file "utils/logging.py" "Logging setup with colored console output."
create_py_file "utils/run_utils.py" "Saving/plotting metrics dictionary history."

# Scripts (Mark as executable later)
create_py_file "scripts/run_training.py" "Unified script to launch training (PT or MLX)."
create_py_file "scripts/run_evaluation.py" "Unified script to launch evaluation."
create_py_file "scripts/run_inference.py" "Unified script for generating captions/answers."
create_py_file "scripts/extract_features.py" "(Optional) Script to pre-compute image features."
create_py_file "scripts/generate_project_doc.py" "(Optional) Utility to generate project documentation."
create_py_file "scripts/setup_project.py" "(Optional) Script to recreate this structure."

echo "  ✅ Core Python files created."
echo "-----------------------------------------------------"

# --- 4. Create Config Files ---
echo "⚙️ Creating Configuration Files..."

# .gitignore
echo "Creating .gitignore..."
cat << EOF > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
# According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
# Pipfile

# PEP 582; used by PDM, PEP 582 proposals
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
venv/
env/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static analysis results
.pytype/

# Cython debug symbols
cython_debug/

# VSCode files
.vscode/

# IDEA files
.idea/

# PyCharm files
*.iml
modules.xml
wsgi.xml

# Data files
data/

# Model files
models/

# Log files
logs/
*.log

# W&B local files
wandb/

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Optional Large Files Cache (e.g. DVC)
*.dvc
.dvc/cache

# Notebook output files
*.ipynb
*.html # If saving notebooks as html
temp_*.png # Temporary plots from tests

# MLX Checkpoints (if different format needed)
# *.safetensors

# Streamlit
.streamlit/
EOF
echo "  📄 Created .gitignore"

# .dockerignore
echo "Creating .dockerignore..."
cat << EOF > .dockerignore
**/.git
**/.venv
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
.DS_Store
.ipynb_checkpoints
notebooks/
tests/
data/ # Usually don't copy data into image
models/ # Usually mount models or download them
logs/
wandb/
*.log
*.pkl
*.pth
*.safetensors
*.png
*.json # Exclude metrics etc. unless needed
*.ipynb
*.md # Exclude docs unless needed
config.yaml # Often mounted as volume or passed via env vars
Dockerfile
.dockerignore
.gitignore
.flake8
pyproject.toml
.pre-commit-config.yaml
scripts/ # Unless needed for entrypoint/setup inside container
*.dvc # If using DVC
.dvc/
EOF
echo "  📄 Created .dockerignore"

# requirements.txt
echo "Creating requirements.txt..."
cat << EOF > requirements.txt
# Core ML Frameworks
torch
torchvision
torchaudio
mlx>=0.6 # Example version constraint
transformers # For loading ViT/CLIP/Tokenizers
datasets # For loading standard datasets

# Utilities
wandb # Experiment tracking
pyyaml # For config.yaml
tqdm # Progress bars
numpy
pillow # Image handling
matplotlib # Plotting
seaborn # Nicer plots
pandas # Optional: data manipulation if needed

# Web App & Deployment
streamlit
streamlit-drawable-canvas # For drawing input

# Evaluation Metrics
nltk # For BLEU score
rouge-score # For ROUGE score
# pycocoevalcap # For METEOR, CIDEr (might need specific install steps)

# Testing
pytest

# Code Quality (Optional for runtime, good for dev)
# black
# flake8
# isort
# mypy
# pre-commit

# Optional: Specific Model Needs
# torchinfo # For model summaries

EOF
echo "  📄 Created requirements.txt"

# Dockerfile
echo "Creating Dockerfile..."
cat << EOF > Dockerfile
# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create final application image
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and necessary utilities
COPY app/ app/
COPY src/ src/
COPY utils/ utils/
# COPY config.yaml . # Might mount this instead

# Download NLTK data if needed for BLEU score
# RUN python -m nltk.downloader punkt

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD streamlit hello

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

EOF
echo "  📄 Created Dockerfile"

# config.yaml
echo "Creating config.yaml..."
cat << EOF > config.yaml
# $PROJECT_NAME - Configuration
# Copyright (c) $COPYRIGHT_YEAR $TEAM_NAME
# File: config.yaml
# Description: Central configuration file.
# Created: $CREATION_DATE
# Updated: $CREATION_DATE

# --- General Paths ---
paths:
  model_save_dir: "models/image_captioning" # Example subdir for task
  log_dir: "logs"
  log_file_name: "multimodal_train.log"
  data_dir: "data" # Default location for downloaded datasets

# --- Tokenizer Configuration ---
tokenizer:
  hf_name: "openai/clip-vit-base-patch32" # Example: Use CLIP's tokenizer
  # Or: "gpt2", "bert-base-uncased" etc.
  pad_token_id: 0 # Check tokenizer.pad_token_id after loading
  start_token_id: 1 # Define custom or use tokenizer.bos_token_id
  end_token_id: 2   # Define custom or use tokenizer.eos_token_id
  vocab_size: 13  # Placeholder - get this from loaded tokenizer
  max_seq_len: 50 # Max caption length to generate/pad to

# --- Dataset Parameters ---
dataset:
  name: "flickr30k" # Example: 'flickr30k', 'coco_captions' (use HF dataset name)
  image_size: 224    # Input size expected by ViT/CLIP
  patch_size: 16     # Example for ViT-Base-Patch16
  # Phase specific params not strictly needed if handling via task/model config

# --- Vision Encoder Configuration ---
encoder:
  hf_name: "google/vit-base-patch16-224-in21k" # Example ViT
  # Or: "openai/clip-vit-base-patch32" # Example CLIP vision tower
  output_layer: -1 # Which layer's features to extract (-1 is usually last hidden state)
  # Add flags for freezing etc. if needed

# --- Decoder Model Hyperparameters ---
model: # Parameters for the custom decoder
  embed_dim: 768       # Should match encoder output dimension if directly connected
  depth: 6             # Number of decoder blocks
  num_heads: 8         # Number of attention heads
  mlp_ratio: 4.0       # Decoder MLP expansion ratio
  dropout: 0.1
  attention_dropout: 0.1

# --- Training Hyperparameters ---
training:
  # Define ONE set of params initially, can add phase-specific later if needed
  epochs: 20
  batch_size: 64
  base_lr: 5e-5       # Lower LR common for fine-tuning / transfer learning
  weight_decay: 0.01
  warmup_steps: 500    # Example: Use steps instead of epochs for warmup
  optimizer: "AdamW"
  scheduler: "CosineDecayWithWarmup" # Example
  gradient_clipping: 1.0
  # num_train_samples: -1 # Use -1 for full dataset, or set number
  # num_val_samples: -1
  save_every: 1        # Save checkpoint every N epochs
  eval_every: 1        # Evaluate every N epochs

# --- Evaluation Parameters ---
evaluation:
  batch_size: 128        # Can often be larger than training batch size
  # Add metric calculation settings if needed

# --- Inference Parameters ---
inference:
  generation_method: "greedy" # or "beam"
  beam_size: 5              # if using beam search
  max_gen_len: 50           # Max tokens to generate

# --- Logging Configuration ---
logging:
  log_level: "INFO"
  log_file_enabled: True
  log_console_enabled: True
  log_max_bytes: 10485760
  log_backup_count: 5
EOF
echo "  📄 Created config.yaml"

# README.md
echo "Creating README.md..."
cat << EOF > README.md
# $PROJECT_NAME 🚀

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team $TEAM_NAME** for Week 4 of the MLX Institute Intensive Program.

## Project Overview 📋

Implement Multimodal Transfer Learning, likely focusing on Image Captioning. This involves using a pre-trained Vision Encoder (ViT/CLIP) and building a custom Transformer Decoder (in PyTorch and MLX) to generate text descriptions for images. The project includes deployment via Streamlit and Docker.

*(More details to be added)*

## Directory Structure 📁

Refer to \`docs/STRUCTURE.MD\`.

## Setup 💻

1. Clone repository.
2. Create virtual environment: \`$PYTHON_CMD -m venv $VENV_NAME\`
3. Activate environment: \`source $VENV_NAME/bin/activate\`
4. Install dependencies: \`pip install -r requirements.txt\`
5. Log in to W&B: \`wandb login\`
6. (Optional) Download data explicitly: \`python scripts/download_data.py\` (if script created)

## Usage 🚦

*   **Training:** \`python scripts/run_training.py --framework <mlx|pytorch> [options...]\`
*   **Evaluation:** \`python scripts/run_evaluation.py --run-dir <path_to_model_run> [--framework <mlx|pytorch>]\`
*   **Inference:** \`python scripts/run_inference.py --image-path <path_to_image> --run-dir <path_to_model_run> [--framework <mlx|pytorch>]\`
*   **Web App:** \`streamlit run app/app.py\`

*(More details to be added)*

EOF
echo "  📄 Created README.md"

# pyproject.toml
echo "Creating pyproject.toml..."
cat << EOF > pyproject.toml
[tool.black]
line-length = 79
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 79

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true # Be careful with this

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q"
testpaths = [
    "tests",
]
EOF
echo "  📄 Created pyproject.toml"

# .flake8
echo "Creating .flake8..."
cat << EOF > .flake8
[flake8]
max-line-length = 79
extend-ignore = E203, W503 # Ignore whitespace before ':' (black conflict) and line break before binary operator
exclude = .git,__pycache__,.venv,venv,build,dist,docs/conf.py
max-complexity = 10
EOF
echo "  📄 Created .flake8"

# .pre-commit-config.yaml
echo "Creating .pre-commit-config.yaml..."
cat << EOF > .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0 # Use latest tag
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
-   repo: https://github.com/psf/black
    rev: 23.11.0 # Use latest tag or pinned version
    hooks:
    -   id: black
        args: [--line-length=79]
-   repo: https://github.com/pycqa/isort
    rev: 5.12.0 # Use latest tag or pinned version
    hooks:
      - id: isort
        name: isort (python)
        args: ["--profile", "black", "--line-length=79"]
-   repo: https://github.com/pycqa/flake8
    rev: 6.1.0 # Use latest tag or pinned version
    hooks:
    -   id: flake8
EOF
echo "  📄 Created .pre-commit-config.yaml"

echo "  ✅ Config files created."
echo "-----------------------------------------------------"

# --- 5. Initialize Git ---
echo "🔄 Initializing Git repository..."
git init
git add .
git commit -m "Initial project structure setup"
echo "  ✅ Git repository initialized and initial commit made."
echo "-----------------------------------------------------"

# --- 6. Create Virtual Environment ---
echo "🐍 Creating Python virtual environment ($VENV_NAME)..."
$PYTHON_CMD -m venv $VENV_NAME
echo "  ✅ Virtual environment created."
echo "-----------------------------------------------------"

# --- 7. Set Script Permissions ---
echo "🔑 Setting script permissions..."
chmod +x scripts/*.py || true # Allow failure if no files match yet
chmod +x setup_project.sh
echo "  ✅ Script permissions set."
echo "-----------------------------------------------------"

# --- Done ---
echo "🎉 Project setup complete! 🎉"
echo ""
echo "Next Steps:"
echo "1. Activate the virtual environment: source $VENV_NAME/bin/activate"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. (Optional) Install pre-commit hooks: pre-commit install"
echo "4. Start developing!"
echo "-----------------------------------------------------"