# TumorDetect AI

An advanced diagnostic application that analyzes MRI scans to identify and classify brain tumors, providing radiologists with accurate insights and reducing interpretation time.

## Overview

TumorDetect AI leverages a TensorFlow-based deep learning Convolutional Neural Network (CNN) model to classify brain tumors from MRI images. The integration of artificial intelligence (AI) and machine learning (ML) in medical imaging aims to enhance diagnostic accuracy and efficiency, reducing reliance on manual interpretation by radiologists.

## Features

- **MRI Image Analysis**: Upload and analyze brain MRI scans
- **Tumor Classification**: Identifies four categories of results:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor (Normal)
- **Confidence Scoring**: Provides confidence level for each prediction
- **User-Friendly Interface**: Simple web interface for easy interaction
- **Detailed Results**: Offers information about detected tumor types

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: PIL, NumPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/TumorDetect.git
   cd TumorDetect
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## Model Architecture

The CNN model consists of multiple convolutional blocks with batch normalization, max pooling, and dropout layers to prevent overfitting. The architecture is designed to effectively extract features from MRI images and classify them into different tumor categories.

## Dataset

The model is trained on a dataset of brain MRI scans containing images of different tumor types and normal brain tissue. The dataset is organized into four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Usage

1. Navigate to the home page
2. Upload an MRI scan image (JPG, JPEG, or PNG format)
3. Click "Analyze Image"
4. View the results, including:
   - Tumor classification
   - Confidence level
   - Information about the detected tumor type

## Disclaimer

TumorDetect AI is designed to assist medical professionals, not replace them. All results should be verified by qualified healthcare providers. This tool is not intended for self-diagnosis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Medical professionals who provided domain expertise
- Open source community for libraries and tools