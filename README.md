# DeepSam

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pypi-DeepSam-lightgrey.svg)](#) <!-- replace if published -->

DeepSam is a modular toolbox that integrates modern segmentation models (e.g., Segment Anything / SAM) with deep learning workflows for image segmentation, annotation, and research experimentation. It provides a lightweight API and CLI for inference, tools for dataset conversion, and utilities to train and evaluate custom segmentation models.

Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
  - [Install](#install)
  - [Run inference (Python)](#run-inference-python)
  - [Run inference (CLI)](#run-inference-cli)
- [Training](#training)
- [Dataset Format](#dataset-format)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

Features
- Easy-to-use Python API for segmentation inference and post-processing
- Command-line tools for single-image and batch processing
- Utilities to convert common dataset formats (COCO, VOC) to the project's format
- Hooks for plugging in different backbone models and samplers
- Example scripts for training and evaluation workflows
- Preconfigured configs for quick prototyping

Quick Start

Install
1. Clone the repo:
   git clone https://github.com/raghulchandramouli/DeepSam.git
   cd DeepSam

2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. (Optional) Install in editable mode:
   pip install -e .

Run inference (Python)
Below is a minimal example showing how to run inference using the Python API.

Run inference (CLI)
The repository includes a simple CLI for inference on single images or whole directories.

Single image:
   python scripts/infer.py --image examples/images/example.jpg --checkpoint path/to/checkpoint.pth --out-dir outputs/

Batch inference:
   python scripts/infer.py --input-dir examples/images/ --checkpoint path/to/checkpoint.pth --out-dir outputs/ --batch-size 8

Training
Training scripts are provided for experimentation. The following is a high-level example — consult the actual training script and config files in `configs/` for exact options.

1. Prepare your dataset in the expected format (see [Dataset Format](#dataset-format)).
2. Launch training:
   python scripts/train.py --config configs/deepsam_default.yaml --work-dir runs/deepsam_experiment

Training flags you may commonly use:
- --resume-from path/to/checkpoint.pth
- --gpus 0,1  (or use CUDA_VISIBLE_DEVICES)
- --batch-size 16
- --max-epochs 50

Dataset Format
DeepSam supports common dataset formats; conversion utilities are provided.

- COCO-like: JSON annotations with polygons/rle; images in folder.
- VOC-like: per-image mask PNGs with instance ids or class ids.
- Custom: folders with images and a parallel folder of mask files (one mask per instance or semantic mask)

Use the provided converter script to migrate datasets:
   python scripts/convert_dataset.py --src-format coco --src-path /path/to/annotations.json --dst-path data/deepsam/

Evaluation
Evaluation scripts compute common segmentation metrics:
- mIoU (mean Intersection over Union)
- Dice / F1 score
- Precision / Recall at pixel and instance levels

Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch: git checkout -b feature/your-feature
3. Make changes and add tests if appropriate.
4. Run tests and linters:
   pytest
   flake8 deepsam
5. Submit a pull request describing your change.

Please open an issue for larger design or API changes before implementing them so we can discuss the best approach.

Citation
If you use DeepSam in your research, please cite the project (bibliographic entry placeholder):

```
@misc{deepsam2025,
  title = {DeepSam: A toolkit for segmentation with SAM},
  author = {Your Name and Contributors},
  year = {2025},
  howpublished = {\\url{https://github.com/raghulchandramouli/DeepSam}}
}
```

License
This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

Contact
Project lead: raghulchandramouli
For questions or support, open an issue in this repository or reach out via GitHub Discussions (if enabled).

Acknowledgements
- Segment Anything (SAM) model and research from Meta AI
- Community contributors and open-source libraries (PyTorch, torchvision, etc.)

Notes / Next steps
- Replace placeholder paths and commands with the exact script names and checkpoints present in the repository.
- Add badges for CI, code coverage, and package status when available.
- Include prebuilt Dockerfile or example Colab notebooks for easier onboarding.
