# Leaf-Disease-Detection
## Project Overview
  The goal of this project is to develop a deep learning model to detect diseases in leaves based on images. The project involves designing a deep learning architecture, building a user interface using Streamlit, and deploying the application. This README provides a detailed guide to understand, set up, and use the project.

## Table of Contents
  1. Dataset Acquisition
  2. Data Preprocessing
  3. Model Building
  4. Model Evaluation
  5. Application Interface
  6. Deployment
  7. Documentation
  8. How to Run
  9. License
  10. Acknowledgements
## Dataset Acquisition
* Dataset Source: https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset
* Description: The dataset contains images of leaves labeled with different types of diseases. Download the dataset and place it in the data directory.
## Data Preprocessing
### Steps
1. Data Cleaning:

* Removed any corrupted or incomplete images.
* Addressed class imbalance if necessary.
2. Data Augmentation:

* Applied transformations such as rotation, scaling, and flipping to enhance the training dataset while preserving the original images' color information.
* Utilized libraries like TensorFlow or PyTorch for augmentation.
3. Normalization:
* Normalized image pixel values to a range of [0, 1] for consistent model input.
### Scripts
* Preprocessing Script: preprocessing.py
## Model Building
### Architecture
* Designed a deep learning model using Convolutional Neural Networks (CNN).
* Optional: Used pretrained models such as ResNet or EfficientNet for transfer learning.
### Training
* Training Script: train_model.py
* Hyperparameters tuned: Learning rate, batch size, number of epochs.
* Split dataset into training and validation sets for model evaluation during training.
## Model Evaluation
### Metrics
* Accuracy: Evaluated the overall accuracy of the model.
* Precision, Recall, F1 Score: Calculated these metrics for each class to assess model
  performance comprehensively.
* Confusion Matrix: Visualized model predictions versus actual labels.
### Evaluation Script
* Evaluation Script: evaluate_model.py
## Application Interface
### Streamlit App
* App Script: app.py
*Features:
    * Upload leaf images and get disease predictions.
    * Display the uploaded image alongside the predicted disease class and probability.
### Installation and Usage
1. Install Dependencies:
   
       pip install -r requirements.txt
2. Run Streamlit App:

       streamlit run app.py
## Deployment
* Platform: Deployed on Streamlit Sharing, Heroku, or AWS.
* Application Link: Deployed Application
## Documentation
* Preprocessing: preprocessing.py - Details the data cleaning and augmentation process.
* Model Training: train_model.py - Describes the model architecture, training process, and hyperparameter tuning.
* Model Evaluation: evaluate_model.py - Contains code for evaluating the model performance.
* Streamlit App: app.py - Implements the user interface for the web application.
## How to Run
1.Clone the Repository:

    git clone https://github.com/your-username/leaf-disease-detection.git
    cd leaf-disease-detection

2. Install Dependencies:
   
       pip install -r requirements.txt

3. Run Data Preprocessing:

       python preprocessing.py

4. Train the Model:

       python train_model.py

5. Evaluate the Model:

       python evaluate_model.py

6. Start the Streamlit App:

       streamlit run app.py
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
* Dataset: https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset
* Libraries: TensorFlow, PyTorch, scikit-learn, Streamlit
