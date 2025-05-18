import os
import numpy as np
import tensorflow as tf
from models.tumor_model import create_model, train_model, load_model
from data.data_utils import preprocess_image_file
import matplotlib.pyplot as plt

def initialize_model():
    """
    Initialize a new model or load an existing one
    
    Returns:
        Initialized model
    """
    model_path = 'models/brain_tumor_model.h5'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(model_path)
    else:
        print("Creating new model")
        model = create_model()
        # Save the untrained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Saved initial model to {model_path}")
    
    # Print model summary
    model.summary()
    
    return model

def test_prediction(model, image_path=None):
    """
    Test the model with a sample image
    
    Args:
        model: Loaded model
        image_path: Path to test image (optional)
    """
    # Use a sample image if none provided
    if image_path is None or not os.path.exists(image_path):
        print("No valid image path provided. Using a random test image.")
        # Create a random test image (this is just for testing the pipeline)
        test_img = np.random.rand(224, 224, 3)
        plt.imsave('test_image.jpg', test_img)
        image_path = 'test_image.jpg'
    
    # Preprocess the image
    processed_image = preprocess_image_file(image_path)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Class names
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"All class probabilities: {prediction[0]}")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Initialize the model
    model = initialize_model()
    
    # Test with a sample image if available
    test_prediction(model)
    
    print("\nModel is ready for use in the application.")
    print("To train the model with your own data, use the train_model function from models.tumor_model.")