# Multimodal Transfer Learning / Image Captioning 🚀

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)** for Week 4 of the MLX Institute Intensive Program.

## Project Overview 📋

Implement Multimodal Transfer Learning, likely focusing on Image Captioning. This involves using a pre-trained Vision Encoder (ViT/CLIP) and building a custom Transformer Decoder (in PyTorch and MLX) to generate text descriptions for images. The project includes deployment via Streamlit and Docker.

*(More details to be added)*

## Directory Structure 📁

Refer to `docs/STRUCTURE.MD`.

## Setup 💻

1. Clone repository.
2. Create virtual environment: `python3 -m venv .venv`
3. Activate environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Log in to W&B: `wandb login`
6. (Optional) Download data explicitly: `python scripts/download_data.py` (if script created)

## Usage 🚦

*   **Training:** `python scripts/run_training.py --framework <mlx|pytorch> [options...]`
*   **Evaluation:** `python scripts/run_evaluation.py --run-dir <path_to_model_run> [--framework <mlx|pytorch>]`
*   **Inference:** `python scripts/run_inference.py --image-path <path_to_image> --run-dir <path_to_model_run> [--framework <mlx|pytorch>]`
*   **Web App:** `streamlit run app/app.py`

*(More details to be added)*
