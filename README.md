# Facial Recognition Classifier using VGG16 and TensorFlow

**Google Colab Implementation**

---

## Overview

This project implements a facial recognition system using **TensorFlow** and **Keras**, leveraging **transfer learning** with the pre-trained **VGG16** architecture.
The model is designed to classify celebrity faces and was fully developed and executed in **Google Colab**.

The dataset consists of **6 celebrity classes**, each containing approximately **100 images**, resulting in a total of **600 facial images**.

---

## Features

* Fully executable in Google Colab
* Custom facial image dataset (600 images, 6 classes)
* Manual 70/30 training and testing split
* Image preprocessing and label one-hot encoding
* Transfer learning using pre-trained VGG16 (ImageNet)
* Fine-tuning of upper layers for improved performance
* Training and validation accuracy/loss visualization
* Prediction on unseen face images
* Model export in `.h5` format

---

## Model Architecture

* **Base Model:** VGG16

  * Pre-trained on ImageNet
  * Top layers removed
* **Custom Classification Head:**

  * Flatten
  * Dense (64) → Batch Normalization → Dropout
  * Dense (32) → Batch Normalization
  * Dense (16) → Batch Normalization
  * Dense (6) with `softmax` activation

---

## Dataset Structure

The dataset is organized into folders, where each folder name represents a class label:

```text
dataset.zip
└── dataset/
    ├── Tom Cruise/
    ├── Scarlett Johansson/
    ├── Will Smith/
    ├── Leonardo DiCaprio/
    ├── Johnny Depp/
    └── Megan Fox/
```

* Each class contains approximately 100 images
* Upload `dataset.zip` to Google Colab when prompted

---

## How to Run (Google Colab)

1. Upload `dataset.zip` using the Colab file upload prompt
2. Run all notebook cells to:

   * Extract the dataset
   * Preprocess images
   * Train and evaluate the model
3. Modify the `img_path` variable to test predictions on new images
4. Download the trained model (`vgg16_finetuned_model.h5`) if required

---

## Results

* **Initial Training Epochs:** 2
* **Fine-Tuning Epochs:** 5
* **Validation Accuracy:** Approximately 85%–95% (dependent on image quality)

---

## Example Prediction

```text
Predicted class: Leonardo DiCaprio
```

---

## Requirements

Install dependencies in Google Colab:

```bash
pip install tensorflow pandas
```

---

## Future Improvements

* Add data augmentation to improve generalization
* Improve class balance and dataset size
* Deploy the model using Streamlit or Flask
* Integrate webcam support for real-time face recognition

---

## Project Files

* `facial_recognition_colab.ipynb` — Complete Google Colab notebook
* `vgg16_finetuned_model.h5` — Trained Keras model
* `README.md` — Project documentation

---

## Authors : 
Omar EL Gohary
- **LinkedIn:** [linkedin.com/in/omarelgohary2003](https://www.linkedin.com/in/omarelgohary2003/)
* **Email:** [omarrmgohary@gmail.com](mailto:omarrmgohary@gmail.com)

Youssef Azmy
- **LinkedIn:** [linkedin.com/in/omarelgohary2003](https://www.linkedin.com/in/youssef-azmy/)

Hazem Osama
- **LinkedIn:** [linkedin.com/in/hazem-osama25/](https://www.linkedin.com/in/hazem-osama25/)

Youssef Mohamed Shaheen

## License

This project is licensed under the **MIT License**.

