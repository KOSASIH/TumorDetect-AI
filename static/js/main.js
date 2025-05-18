document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileUpload = document.getElementById('file-upload');
    const loadingIndicator = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const uploadedImage = document.getElementById('uploaded-image');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const tumorInfo = document.getElementById('tumor-info');
    
    // Tumor information for different types
    const tumorTypes = {
        'Glioma': {
            description: 'Gliomas are tumors that occur in the brain and spinal cord. They begin in the glial cells that surround and support nerve cells.',
            characteristics: 'Often appears as irregular, poorly defined masses with varying enhancement patterns.'
        },
        'Meningioma': {
            description: 'Meningiomas are tumors that arise from the meninges — the membranes that surround your brain and spinal cord.',
            characteristics: 'Usually appears as well-defined, extra-axial masses with homogeneous enhancement.'
        },
        'No Tumor': {
            description: 'No tumor detected in the brain tissue.',
            characteristics: 'Normal brain structure without abnormal growth or mass effect.'
        },
        'Pituitary': {
            description: 'Pituitary tumors are abnormal growths that develop in the pituitary gland at the base of the brain.',
            characteristics: 'Typically appears as a well-defined sellar/suprasellar mass with variable enhancement.'
        }
    };
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileUpload.files[0];
        if (!file) {
            alert('Please select an image file');
            return;
        }
        
        // Check file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (JPEG, JPG, or PNG)');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send request to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display the uploaded image
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            // Display the prediction result
            if (data.success) {
                const prediction = data.prediction;
                const confidence = data.confidence * 100;
                
                // Set prediction text
                predictionResult.textContent = `Prediction: ${prediction}`;
                
                // Set confidence bar
                confidenceBar.style.width = `${confidence}%`;
                confidenceText.textContent = `Confidence: ${confidence.toFixed(2)}%`;
                
                // Set confidence bar color
                if (confidence >= 80) {
                    confidenceBar.className = 'progress-bar bg-success';
                    confidenceText.className = 'high-confidence';
                } else if (confidence >= 60) {
                    confidenceBar.className = 'progress-bar bg-warning';
                    confidenceText.className = 'medium-confidence';
                } else {
                    confidenceBar.className = 'progress-bar bg-danger';
                    confidenceText.className = 'low-confidence';
                }
                
                // Display tumor information
                if (tumorTypes[prediction]) {
                    const info = tumorTypes[prediction];
                    tumorInfo.innerHTML = `
                        <div class="mt-3">
                            <h6 class="tumor-type">${prediction}</h6>
                            <p>${info.description}</p>
                            <p><strong>Characteristics:</strong> ${info.characteristics}</p>
                        </div>
                    `;
                } else {
                    tumorInfo.innerHTML = '';
                }
            } else {
                predictionResult.textContent = 'Analysis failed: ' + (data.message || 'Unknown error');
                confidenceBar.style.width = '0%';
                confidenceText.textContent = '';
                tumorInfo.innerHTML = '';
            }
            
            // Show result container
            resultContainer.classList.remove('d-none');
        })
        .catch(error => {
            loadingIndicator.classList.add('d-none');
            alert('Error: ' + error.message);
        });
    });
    
    // Preview image on file selection
    fileUpload.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
});