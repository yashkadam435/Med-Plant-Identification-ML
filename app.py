# %%
#Requirements
#load med-detect-log and indian medicinal plants in dataset and medicinal_plant_detection_model in models in kaggle 

# %%
# !pip install transformers
# !pip install torchvision

# %%
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)
import cv2

# %%
import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA (GPU) is available!")
else:
    print("CUDA (GPU) is not available. Using CPU.")


# %%
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# %%
import torch

model_path = 'model.pth'

# Attempt to load model on CPU with error handling
try:
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

except Exception as e:
    print("Error loading model:", e)
    # Optionally, you can re-raise the error to get a more detailed traceback
    # raise


# %%
from transformers import pipeline
model_name = "med-detect-log"
pipe = pipeline('image-classification', model=model_name, device=-1)

# %%
image = "alovera.jpg"
img=cv2.imread(image)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

print(pipe(image))

# %%
# !pip install gradio

# %%
# pip install streamlit


import streamlit as st
import cv2
from PIL import Image
from transformers import pipeline

# Load the image classification pipeline
model_name = "med-detect-log"
pipe = pipeline('image-classification', model=model_name, device=-1)

# Function for prediction
def classify_image(image):
    # Get prediction from the model
    predictions = pipe(image)
    return predictions

# Streamlit app
def main():
    st.title("Med-Detect-Log Image Classification")

    # Layout for input section
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Layout for output section
    st.subheader("Output")
    output_placeholder = st.empty()

    if uploaded_file is not None:
        # Read uploaded image
        image = Image.open(uploaded_file)
        # Display input image
        st.subheader("Input Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify image
        predictions = classify_image(image)

        # Display prediction results
        st.subheader("Predictions:")
        for prediction in predictions:
            st.write(f"Label: {prediction['label']}")
            st.write(f"Confidence Score: {prediction['score']:.2f}")
            st.write("---")

        # Add a flag button to store output in a CSV file
        if st.button("Flag for CSV"):
            save_to_csv(predictions)

# Function to save output to CSV
def save_to_csv(predictions):
    import csv
    with open('predictions.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Confidence Score'])
        for prediction in predictions:
            writer.writerow([prediction['label'], prediction['score']])
    st.success("Output saved to predictions.csv")

if __name__ == "__main__":
    main()











# %%
# import streamlit as st
# import cv2
# from transformers import pipeline

# # Load the model pipeline
# model_name = "med-detect-log"
# pipe = pipeline('image-classification', model=model_name, device=-1)

# def main():
#     st.title("Medicinal Plant Identification")

#     st.sidebar.title("Upload Image")
#     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = cv2.imread(uploaded_file.name)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Make prediction
#         predictions = pipe(uploaded_file.name)

#         # Display prediction results
#         st.subheader("Predictions:")
#         for prediction in predictions:
#             st.write(f"Label: {prediction['label']}")
#             st.write(f"Confidence Score: {prediction['score']:.2f}")
#             st.write("---")

# if __name__ == "__main__":
#     main()




