# Automated Medicinal Plant Identification

## Introduction
This repository contains the code and resources for an automated medicinal plant identification system developed using image processing and machine learning techniques. The system aims to accurately identify medicinal plants commonly used in India's Ayurvedic Pharmaceutics, addressing challenges such as morphological variability and market adulteration.

## Features
- **Image Processing:** Utilizes advanced image processing techniques to extract salient features like color, texture, and shape from plant images.
- **Machine Learning:** Integrates machine learning algorithms, including the Xception model, for automated plant identification based on extracted features.
- **Streamlit Web Application:** Includes a user-friendly Streamlit web application for easy access and usage of the model.

## Output

![Output_1](https://github.com/yashkadam435/Med-Plant-Identification-ML/assets/108817280/5be9b55f-5ea8-4e60-b761-2a3eeb320f97)

## Requirements
- Python 3.12.3
- Libraries: Transformers, Torch, Streamlit, OpenCV, PIL

## Installation
1. Clone the repository to your local machine:
   
```bash
git clone https://github.com/your_username/automated-medicinal-plant-identification.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage
1. Ensure that you have installed all the required dependencies.
2. Run the Streamlit web application:

```bash
streamlit run app.py
```

3. Upload an image of a medicinal plant to the web application.
4. The model will process the image and provide predictions on the plant's identity along with confidence scores.

## Model Training
- The model used for plant identification has been trained on a dataset of labelled plant images representing diverse medicinal plants. 
- Training code and dataset details can be found in the `model_training` directory.

## Contributions
Contributions to the repository are welcome! If you have any ideas for improvements or find any issues, feel free to open an issue or submit a pull request.
