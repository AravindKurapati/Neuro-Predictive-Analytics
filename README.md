# Neuro-Predictive-Analytics

## Overview
This project involves classifying harmful brain activities using EEG spectrograms with the help of a deep learning model implemented using KerasCV. The primary goal of this project is to classify different types of brain activities (Seizure, LPD, GPD, etc.) based on EEG data and spectrograms. The project employs various preprocessing techniques, augmentation methods, and model training strategies to achieve this classification task.

## Project Structure
Data Preprocessing: The EEG data is processed to generate spectrograms and stored in .npy format. The dataset is then split into training and validation sets using stratified group K-Folds for a balanced and robust model evaluation.

Model Architecture: The deep learning model is built using KerasCV with the EfficientNetV2-B2 preset, which leverages ImageNet weights. The model is designed to take the processed spectrograms as input and classify them into one of the six predefined classes.

Augmentation: Advanced augmentation techniques such as MixUp and Random Cutout are applied to the spectrograms to increase the robustness of the model.

Training: The model is trained using Adam optimizer with a learning rate scheduler (cosine decay) and KLDivergence loss. The model training is monitored with validation accuracy and loss to prevent overfitting, with the best model checkpoint saved.

Inference: Once trained, the model is used to make predictions on the test dataset, and a submission file is generated in the required format.
