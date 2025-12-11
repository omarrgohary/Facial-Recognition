# Facial Recognition Classifier using VGG16 and TensorFlow (Google Colab)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This project demonstrates a facial recognition system built with TensorFlow and Keras using a transfer learning approach. The model leverages the power of the pre-trained VGG16 architecture to classify celebrity faces. The entire project was developed and executed in Google Colab, using a custom dataset of 6 celebrity classes, each with approximately 100 images (600 total).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Features:

- Runs entirely in Google Colab
- Uses a dataset of 600 facial images (100 per celebrity)
- Manual 70/30 train-test split
- Preprocessing and one-hot encoding of labels
- Transfer learning with pre-trained VGG16
- Fine-tuning of last layers for better performance
- Visualizations of accuracy and loss
- Prediction on new face images
- Model saving (`.h5` format)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Model Architecture:

- Base Model: VGG16 (without top layer, pretrained on ImageNet)
- Custom Layers**:
  - Flatten
  - Dense(64) → BatchNorm → Dropout
  - Dense(32) → BatchNorm
  - Dense(16) → BatchNorm
  - Dense(6) with `softmax` (for classification)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Dataset Structure

The dataset consists of approximately **6,000 face images**, with **1,000 images per celebrity**, organized into folders by class name:

dataset.zip
└── dataset/
├── Tom Cruise/
├── Scarlett Johansson/
├── Will Smith/
├── Leonardo DiCaprio/
├── Johnny Depp/
└── Megan Fox/

yaml
Copy
Edit

Each folder name becomes a class label. Upload the `dataset.zip` file to Colab when prompted.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


How to Run (in Google Colab)

1. Upload `dataset.zip` to Colab using the upload prompt.
2. Run all cells to extract, preprocess, train, and evaluate the model.
3. Modify the `img_path` at the end to predict a new face.
4. Download the fine-tuned model (`vgg16_finetuned_model.h5`) if needed.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Results:

- Initial Training Epochs: 2
- Fine-Tuning Epochs: 5
- Accuracy (sample): 85–95% on validation depending on image quality

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Example Prediction: python

Predicted class: Leonardo DiCaprio
🔧 Requirements
Install packages via Colab:
bash
Copy
Edit
pip install tensorflow pandas

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Future Improvements:

Add data augmentation for better generalization
Improve class balancing if needed
Deploy using Streamlit or Flask
Add webcam support for real-time predictions

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Files:
facial_recognition_colab.ipynb — full Colab-compatible notebook
vgg16_finetuned_model.h5 — saved Keras model
README.md — project documentation

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Author:
Omar El Gohary | Linkedin: linkedin.com/in/omarelgohary2003 
Youssef Azmy   | Linkedin: linkedin.com/in/youssef-azmy/

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

License:
This project is licensed under the MIT License.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
