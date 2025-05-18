#!/usr/bin/env python3
import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import tensorflow as tf
        import flask
        import numpy as np
        import PIL
        print("All core dependencies are installed.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully.")

def initialize_model():
    """Initialize the model"""
    print("Initializing model...")
    try:
        subprocess.check_call([sys.executable, "init_model.py"])
        print("Model initialized successfully.")
    except subprocess.CalledProcessError:
        print("Error initializing model. The application will run without model prediction.")

def create_upload_dir():
    """Create uploads directory if it doesn't exist"""
    os.makedirs("uploads", exist_ok=True)
    print("Uploads directory created.")

def run_app():
    """Run the Flask application"""
    print("Starting TumorDetect AI application...")
    print("The application will be available at http://localhost:8080")
    subprocess.check_call([sys.executable, "app.py"])

if __name__ == "__main__":
    print("=" * 50)
    print("TumorDetect AI - Brain Tumor MRI Analysis")
    print("=" * 50)
    
    # Check if dependencies are installed
    if not check_dependencies():
        install_dependencies()
    
    # Create uploads directory
    create_upload_dir()
    
    # Initialize model
    initialize_model()
    
    # Run the application
    run_app()