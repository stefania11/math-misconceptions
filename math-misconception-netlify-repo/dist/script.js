// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
  const videoElement = document.getElementById('camera-feed');
  const canvasElement = document.getElementById('canvas');
  const captureBtn = document.getElementById('capture-btn');
  const uploadBtn = document.getElementById('upload-btn');
  const fileInput = document.getElementById('file-input');
  const tryAgainBtn = document.getElementById('try-again-btn');
  const cameraView = document.getElementById('camera-view');
  const capturedImageView = document.getElementById('captured-image');
  const cameraPlaceholder = document.getElementById('camera-placeholder');
  const processingStatus = document.getElementById('processing-status');
  const resultSection = document.getElementById('result-section');
  const uploadSection = document.getElementById('upload-section');
  
  let stream = null;
  
  // Start webcam
  async function startCamera() {
    try {
      // Always show the upload section regardless of camera status
      if (uploadSection) {
        uploadSection.style.display = 'block';
      }
      
      // Show camera elements
      if (cameraView) {
        cameraView.style.display = 'block';
      }
      
      if (cameraPlaceholder) {
        cameraPlaceholder.style.display = 'none';
      }
      
      if (videoElement) {
        videoElement.style.display = 'block';
      }
      
      // Get user media
      stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      
      // Connect stream to video element
      if (videoElement) {
        videoElement.srcObject = stream;
      }
      
      // Show capture button
      if (captureBtn) {
        captureBtn.style.display = 'flex';
      }
      
      // Hide try again button
      if (tryAgainBtn) {
        tryAgainBtn.style.display = 'none';
      }
      
      console.log('Camera started successfully');
    } catch (err) {
      console.error('Error starting camera:', err);
      showCameraError();
    }
  }
  
  // Show camera error and fallback to upload
  function showCameraError() {
    if (cameraPlaceholder) {
      cameraPlaceholder.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path><circle cx="12" cy="13" r="3"></circle></svg>
        <p>Camera access denied or not available</p>
        <p style="font-size: 0.875rem;">Please use the upload option below</p>
      `;
      cameraPlaceholder.style.display = 'flex';
    }
    
    if (videoElement) {
      videoElement.style.display = 'none';
    }
    
    if (captureBtn) {
      captureBtn.style.display = 'none';
    }
    
    if (uploadSection) {
      uploadSection.style.display = 'block';
    }
  }
  
  // Capture image from video
  function captureImage() {
    if (!videoElement || !canvasElement) return;
    
    const context = canvasElement.getContext('2d');
    if (!context) return;
    
    // Set canvas dimensions to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    // Get image data URL
    const imageDataUrl = canvasElement.toDataURL('image/jpeg');
    
    // Display captured image
    displayCapturedImage(imageDataUrl);
    
    // Stop camera stream
    stopCamera();
    
    // Process the image
    processImage(imageDataUrl);
  }
  
  // Display captured image
  function displayCapturedImage(imageUrl) {
    if (!capturedImageView) return;
    
    // Hide video element
    if (videoElement) {
      videoElement.style.display = 'none';
    }
    
    // Hide camera placeholder
    if (cameraPlaceholder) {
      cameraPlaceholder.style.display = 'none';
    }
    
    // Show captured image
    capturedImageView.src = imageUrl;
    capturedImageView.style.display = 'block';
    
    // Hide capture button, show try again button
    if (captureBtn) {
      captureBtn.style.display = 'none';
    }
    
    if (tryAgainBtn) {
      tryAgainBtn.style.display = 'flex';
    }
  }
  
  // Handle file upload
  function handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      const imageDataUrl = e.target?.result;
      if (typeof imageDataUrl === 'string') {
        displayCapturedImage(imageDataUrl);
        processImage(imageDataUrl);
      }
    };
    reader.readAsDataURL(file);
  }
  
  // Process the image (simulate API call)
  function processImage(imageDataUrl) {
    if (!processingStatus) return;
    
    // Show processing status
    processingStatus.textContent = 'Analyzing math work...';
    
    // Simulate API call delay
    setTimeout(() => {
      // Show result
      if (resultSection) {
        resultSection.style.display = 'block';
      }
      
      // Update status
      processingStatus.textContent = 'Analysis complete';
      
      // Add to history (in a real app, this would use the actual result)
      addToHistory({
        id: 'MaE44',
        description: 'Struggling with independent/dependent variables',
        timestamp: new Date().toLocaleTimeString()
      });
    }, 2000);
  }
  
  // Add item to history
  function addToHistory(item) {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
      <div>
        <span class="history-id">${item.id}:</span> ${item.description}
      </div>
      <span class="history-time">${item.timestamp}</span>
    `;
    
    historyList.prepend(historyItem);
    
    // Show history section
    const historySection = document.getElementById('history-section');
    if (historySection) {
      historySection.style.display = 'block';
    }
  }
  
  // Stop camera stream
  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
  }
  
  // Reset to initial state
  function resetCamera() {
    // Hide captured image
    if (capturedImageView) {
      capturedImageView.style.display = 'none';
    }
    
    // Hide result section
    if (resultSection) {
      resultSection.style.display = 'none';
    }
    
    // Reset processing status
    if (processingStatus) {
      processingStatus.textContent = '';
    }
    
    // Start camera again
    startCamera();
  }
  
  // Event listeners
  if (captureBtn) {
    captureBtn.addEventListener('click', captureImage);
  }
  
  if (fileInput) {
    fileInput.addEventListener('change', handleFileUpload);
  }
  
  if (uploadBtn) {
    uploadBtn.addEventListener('click', () => {
      if (fileInput) {
        fileInput.click();
      }
    });
  }
  
  if (tryAgainBtn) {
    tryAgainBtn.addEventListener('click', resetCamera);
  }
  
  // Initialize camera on page load
  startCamera();
  
  // Clean up on page unload
  window.addEventListener('beforeunload', stopCamera);
});
