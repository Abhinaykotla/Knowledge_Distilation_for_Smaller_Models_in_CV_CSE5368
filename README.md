# Intel Image Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying images from the Intel Image Classification dataset. The model is designed to be modular, allowing users to specify the number of residual blocks and fully connected layers, ensuring that input and output dimensions are correctly matched.

## Project Structure
```
intel-image-classification
├── data
│   ├── __init__.py
│   └── dataset.py
├── models
│   ├── __init__.py
│   ├── blocks.py
│   └── cnn_model.py
├── utils
│   ├── __init__.py
│   ├── train_utils.py
│   └── visualization.py
├── config.py
├── train.py
├── evaluate.py
├── README.md
└── requirements.txt
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset
The dataset used in this project is the Intel Image Classification dataset, which consists of various scenes and natural objects. The dataset is divided into training, testing, and prediction sets.

## Model Architecture
The CNN model is built using modular components:
- **Residual Blocks**: Configurable blocks that allow for the creation of deep networks while maintaining performance through skip connections.
- **Fully Connected Layers**: Users can specify the number of layers, ensuring that the model can adapt to different classification tasks.

## Usage
To train the model, run the following command:

```bash
python train.py
```

To evaluate the trained model, use:

```bash
python evaluate.py
```

## Visualization
Training results, including loss and accuracy curves, can be visualized using the functions provided in the `utils/visualization.py` module.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.