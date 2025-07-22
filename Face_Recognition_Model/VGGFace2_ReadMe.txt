
VGGFace2 Facial Recognition Model

This repository contains scripts and notebooks to train, validate, and test a facial recognition model using the VGG19 architecture on the VGGFace2 dataset. The project focuses on building a robust face classification model with high accuracy and flexibility for further applications.

Directory Structure:
--------------------
├── VGGFace2Model.ipynb         # Main model training notebook
├── splitTrainVal.ipynb         # Splits dataset into train/val sets
├── mergeTrainVal.ipynb         # Merges separate folders back
├── Testing_VGG.ipynb           # Runs predictions and evaluates the trained model
├── saved_models/               # Folder to store `.h5` model files
├── dataset/                    # Base dataset (VGGFace2)
│   ├── train/
│   ├── val/
│   └── test/

Model Overview:
---------------
- Architecture: VGG19 (pretrained on ImageNet)
- Dataset: VGGFace2
- Classes Used: 300 identities
- Images/Identity: 150 per identity
- Input Size: 224x224
- Framework: TensorFlow/Keras

Environment Requirements:
--------------------------
python>=3.7
tensorflow==2.4.1
numpy
opencv-python
matplotlib
sklearn

Install dependencies:
pip install -r requirements.txt

Data Preparation:
-----------------
Step 1: Split Train and Validation
Run splitTrainVal.ipynb
- Splits original training data into `train` and `val` folders (typically 80/20 split)

Step 2: Merge Data (Optional)
Run mergeTrainVal.ipynb
- Merges the data back into a single folder

Model Training:
---------------
Run VGGFace2Model.ipynb

Key Features:
- Loads VGG19 with include_top=False
- Adds custom classification head
- Freezes and unfreezes layers for fine-tuning
- Uses ImageDataGenerator with data augmentation
- Saves best model as .h5 in saved_models/

Important Hyperparameters:
- Epochs: 30
- Batch size: 32
- Optimizer: Adam
- Loss: categorical_crossentropy
- Metrics: accuracy

Testing & Evaluation:
---------------------
Run Testing_VGG.ipynb

- Loads test images from test/ directory
- Uses the trained model to predict the identity
- Displays sample predictions with ground truth
- Calculates overall accuracy

Sample Output:
--------------
- Classification accuracy: ~XX% (based on your final test results)
- Confusion matrix and misclassified faces shown
- Model inference time per image

Notes:
------
Dataset format:
dataset/
├── train/
│   └── <person_id>/
│       └── *.jpg
├── val/
├── test/

Adjust the number of classes based on your VGGFace2 subset.

Future Work:
------------
- Convert model to .tflite for mobile
- Add face detection pre-processing (e.g., MTCNN)
- Add webcam-based real-time recognition

Contact:
--------
For questions or suggestions, feel free to reach out.
