
# Knowledge Distillation for Efficient Image Classification

This project explores **parameter reduction** and **precision quantization** in image classification models through **knowledge distillation**.  
It demonstrates how student models can be trained to replicate the performance of a larger teacher model while significantly reducing computational requirements.

The implementation is based on the **Intel Image Classification Dataset** and utilizes custom CNN architectures with residual blocks.

## Project Structure
```
Knowledge_Distilation_for_Smaller_Models_in_CV_CSE5368
│   .gitignore
│   config.py
│   evaluate.py
│   plot_experiment_results.py
│   README.md
│   requirements.txt
│   run_experiments.py
│   teacher_model.ipynb
│   train.py
│
├───checkpoints
│
├───data
│   │   dataset.py
│   │
│   └───intel-image-classification
├───Final Paper
│   └───tex
│
├───models
│       blocks.py
│       cnn_model.py
│       teacher_arch.py
│
└───utils
        train_utils.py
        visualization.py
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Abhinaykotla/Knowledge_Distilation_for_Smaller_Models_in_CV_CSE5368
cd Knowledge_Distilation_for_Smaller_Models_in_CV_CSE5368
```

### 2. Install dependencies
Create a Python environment (recommended) and install required libraries:

```bash
pip install -r requirements.txt
```

**Note:** Make sure you are using a CUDA-capable device (NVIDIA GPU) if you want to leverage mixed precision or faster training.

### 3. Download the Intel Image Dataset
- [Download the dataset from Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
- Extract it into the following structure:

```
data/
└── intel-image-classification/
    ├── seg_train/
    ├── seg_test/
    ├── seg_pred/
```

No changes to the folder names are necessary.

## How to Train a Single Model

You can train a student model manually by running:

```bash
python train.py
```

Make sure to edit `config.py` if you want to customize:
- Model architecture (residual blocks, FC layers)
- Batch size, learning rate, precision settings
- Paths for saving models and histories

The best model (based on validation accuracy) will be saved under `/checkpoints`.

## How to Run Full Experiments (Weight Reduction + Precision Quantization)

Run the experiment sweep by:

```bash
python run_experiments.py
```

This will:
- Train different student models (4, 6, 8, 10 residual blocks)
- Evaluate at three precision levels: FP32, FP16, FP8
- Save all results in structured folders under `/checkpoints`
- Each folder contains:
  - Trained model weights
  - Training history (loss/accuracy)
  - A summary `.txt` file (with number of parameters, training time, best test accuracy, etc.)

Progress bars will indicate the status of each experiment.

## How to Analyze Results

After running experiments, you can find results in:

```
checkpoints/
├── res4_b224_fp16/
│   ├── model.pth
│   ├── history.pth
│   └── training_summary.txt
├── res6_b224_fp8/
│   └── ...
└── ...
```

Each `training_summary.txt` file includes:
- Model architecture details
- Number of residual blocks
- Number of parameters
- Best achieved accuracy
- Training time
- Precision used (fp16, fp32, or fp8)
- Early stopping information (if triggered)

You can quickly compare across models and precision settings.

## Notes
- **Mixed precision (FP16)** is enabled using PyTorch's `torch.amp.autocast`.
- **FP8 quantization** is handled using the `bitsandbytes` library.
- **Early stopping** with patience of 3 epochs is applied automatically to avoid overfitting.
- The Intel Image Classification Dataset contains **6 classes** (buildings, forest, glacier, mountain, sea, street).

## Future Work
- Extending knowledge distillation to **image inpainting** or **semantic segmentation**.
- Integrating more aggressive quantization (1-bit, 2-bit models).
- Exploring self-distillation or feature-based distillation techniques.

## References
- [PyTorch](https://pytorch.org/)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [Intel Image Classification Dataset on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

This project was developed as part of coursework for **CSE 5368 Neural Networks** at the University of Texas at Arlington.

# Knowledge distillation for better, faster, smaller vision models!
