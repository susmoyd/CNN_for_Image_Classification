# CNN for Image Classification: Dog vs Cat

This project implements a Convolutional Neural Network (CNN) using TensorFlow (integrated with Keras) to classify images as either a dog or a cat. The model is trained on a labeled dataset of 4000 dog images and 4000 cat images, and evaluated using a test dataset of 1000 dog images and 1000 cat images. The final model can also make predictions on new images stored in the `single_prediction` folder.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [How to Run the Project](#how-to-run-the-project)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## Project Overview
The primary objective of this project is to:
1. Build and train a CNN using TensorFlow and Keras.
2. Evaluate the performance of the model using a test dataset.
3. Classify new images as either "Dog" or "Cat" using the trained model.

This project demonstrates the power of deep learning for image classification tasks and serves as a starting point for further exploration of computer vision techniques.

---

## Dataset Description
- **Training Set**: 4000 images of dogs, 4000 images of cats.
- **Test Set**: 1000 images of dogs, 1000 images of cats.
- **Single Prediction Folder**: Contains images to be classified by the trained model.

---

## Model Architecture
The CNN model consists of the following layers:
1. **Convolutional Layers**: Extract features from input images.
2. **Pooling Layers**: Downsample feature maps to reduce spatial dimensions.
3. **Flatten Layer**: Convert feature maps into a single vector.
4. **Fully Connected Layers**: Perform classification using dense layers.
5. **Output Layer**: Sigmoid activation function for binary classification.

The model is compiled with:
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Evaluation Metric**: `accuracy`

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>

2. Install the dependencies:

pip install -r requirements.txt

3. Train the model: Run the Jupyter Notebook CNN for Image Classification.ipynb to train the model on the training dataset and evaluate it on the test dataset.

4. Make predictions: Use the trained model to classify images in the single_prediction folder.

Dependencies
The project requires the following Python libraries:

tensorflow
numpy
matplotlib
Pillow (for image preprocessing)

Install them using the following command:

pip install tensorflow numpy matplotlib Pillow

Project Structure

├── CNN for Image Classification.ipynb  # Jupyter notebook with the implementation

├── single_prediction/                  # Folder containing images for classification

├── datasets/                           # Folder containing training and test datasets

├── README.md                           # Project documentation

└── requirements.txt                    # Python dependencies

Results

Training Accuracy: Achieved high accuracy on the training dataset after several epochs.
Test Accuracy: Successfully generalized to the test dataset, with good classification performance.
Prediction Example: The model can classify new images as either a "Dog" or a "Cat" with high confidence.

License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

Acknowledgments

Special thanks to TensorFlow and Keras for providing powerful tools to build deep learning models, and to the creators of the datasets used in this project.

Medium Link: https://medium.com/@susmoyd21/building-a-cnn-to-classify-cats-and-dogs-a-step-by-step-guide-ba5a7007dbfe

Dataset Link: https://www.dropbox.com/scl/fo/4vh2d4pmoeyikiog4fgk3/ABHU1_Zz63Cxo2xNPPb0IR8?rlkey=5372q4s5spzpblxcqolr10dkq&st=2yb90qp7&dl=0
