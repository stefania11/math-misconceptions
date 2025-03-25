'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Camera, Menu, AlertCircle, Info, BookOpen, RotateCcw, Upload } from 'lucide-react';
import { analyzeMathProblem, MathMisconception } from '@/services/geminiService';

// Define the history item type
interface HistoryItem {
  id: string;
  description: string;
  timestamp: string;
}

const MathMisconceptionDetector: React.FC = () => {
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'complete'>('idle');
  const [detectedMisconception, setDetectedMisconception] = useState<MathMisconception | null>(null);
  const [feedback, setFeedback] = useState<string>('');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Sample misconception database (would connect to your research data)
  const misconceptionDatabase: MathMisconception[] = [
    {
      id: 'MaE07',
      description: 'Simplifying just one term in a fraction',
      example: 'Simplifying only the numerator or denominator in a fraction',
      feedback: 'Remember to apply the same operation to both the numerator and denominator when simplifying fractions.'
    },
    {
      id: 'MaE11',
      description: 'Subtracting mixed numbers incorrectly',
      example: 'Avoiding regrouping and just subtracting smaller from larger',
      feedback: 'When subtracting mixed numbers, you may need to regroup (borrow) when the fraction part of the first number is smaller than the fraction part of the second number.'
    },
    {
      id: 'MaE44',
      description: 'Struggling with independent/dependent variables',
      example: 'Difficulty identifying which variable depends on the other',
      feedback: 'The dependent variable (output) changes based on the independent variable (input). Think about cause and effect - which variable controls the other?'
    },
    {
      id: 'MaE49',
      description: 'Difficulty with variable meanings',
      example: 'Struggling to comprehend various meanings and applications of variables',
      feedback: 'Variables can represent unknown values, changing quantities, or relationships between values. In this problem, try to clearly identify what the variable represents.'
    },
    {
      id: 'MaE53',
      description: 'Confusion with operation symbols',
      example: 'Mixing up operation symbols and their meanings in algebra',
      feedback: 'Be careful with operation symbols. Remember that each symbol (+, -, ×, ÷, =) has a specific meaning in the context of an equation.'
    }
  ];

  // State for camera error
  const [cameraError, setCameraError] = useState<boolean>(false);
  
  // Handle file upload
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageDataUrl = e.target?.result as string;
      setCapturedImage(imageDataUrl);
      setIsCapturing(false);
      analyzeMathWork();
    };
    reader.readAsDataURL(file);
  };
  
  // Start video stream
  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream;
      setIsCapturing(true);
      setProcessingStatus('idle');
      setCapturedImage(null);
      setDetectedMisconception(null);
      setFeedback('');
      setCameraError(false);
    } catch (err) {
      console.error("Error accessing camera:", err);
      setCameraError(true);
    }
  };

  // Stop video stream
  const stopCapture = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      setIsCapturing(false);
    }
  };

  // Capture image from video
  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        setCapturedImage(canvas.toDataURL('image/jpeg'));
        stopCapture();
        analyzeMathWork();
      }
    }
  };

  // Analyze the math work using Gemini Flash 2.0 API
  const analyzeMathWork = async () => {
    if (!capturedImage) return;
    
    setProcessingStatus('processing');
    
    try {
      // Call the Gemini API to analyze the image
      const result = await analyzeMathProblem(capturedImage);
      
      if (result) {
        // Use the detected misconception from Gemini
        setDetectedMisconception(result);
        setFeedback(result.feedback);
        
        // Add to history
        setHistory(prev => [...prev, {
          id: result.id,
          description: result.description,
          timestamp: new Date().toLocaleTimeString()
        }]);
      } else {
        // Fallback to a random misconception if Gemini couldn't detect one
        const randomIndex = Math.floor(Math.random() * misconceptionDatabase.length);
        const fallbackIssue = misconceptionDatabase[randomIndex];
        
        setDetectedMisconception(fallbackIssue);
        setFeedback(fallbackIssue.feedback + " (Note: This is a sample misconception as no specific issue was detected in your work.)");
        
        // Add to history with a note that it's a fallback
        setHistory(prev => [...prev, {
          id: fallbackIssue.id + " (Sample)",
          description: fallbackIssue.description,
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    } catch (error) {
      console.error("Error analyzing math work:", error);
      // Handle error case - maybe show an error message to the user
    } finally {
      setProcessingStatus('complete');
    }
  };

  // Reset the detection
  const resetDetection = () => {
    setCapturedImage(null);
    setProcessingStatus('idle');
    setDetectedMisconception(null);
    setFeedback('');
    startCapture();
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Initialize camera on component mount
  useEffect(() => {
    startCapture();
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">Math Misconception Detector</h1>
          <Menu className="h-6 w-6" />
        </div>
      </header>

      <main className="flex-1 p-4 flex flex-col gap-4">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="relative aspect-video bg-black">
            {isCapturing && !cameraError && (
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                className="w-full h-full object-cover"
              />
            )}
            {cameraError && !capturedImage && (
              <div className="w-full h-full flex items-center justify-center bg-gray-800 text-white">
                <div className="text-center p-4">
                  <p className="mb-4">Camera not available. Please upload an image instead.</p>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-full flex items-center gap-2 mx-auto"
                  >
                    <Upload className="h-5 w-5" /> Upload Image
                  </button>
                  <input 
                    type="file" 
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept="image/*"
                    className="hidden"
                  />
                </div>
              </div>
            )}
            {capturedImage && (
              <img 
                src={capturedImage} 
                alt="Captured math work" 
                className="w-full h-full object-cover"
              />
            )}
            <canvas ref={canvasRef} className="hidden" />
          </div>

          <div className="p-4 flex justify-between items-center">
            {isCapturing && !cameraError ? (
              <button
                onClick={captureImage}
                className="bg-red-500 text-white px-4 py-2 rounded-full flex items-center gap-2"
              >
                <Camera className="h-5 w-5" /> Capture
              </button>
            ) : !capturedImage ? (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-green-500 text-white px-4 py-2 rounded-full flex items-center gap-2"
              >
                <Upload className="h-5 w-5" /> Upload Image
              </button>
            ) : (
              <button
                onClick={resetDetection}
                className="bg-blue-500 text-white px-4 py-2 rounded-full flex items-center gap-2"
              >
                <RotateCcw className="h-5 w-5" /> Try Again
              </button>
            )}
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/*"
              className="hidden"
            />
            <div className="text-sm text-gray-500">
              {processingStatus === 'processing' && 'Analyzing math work...'}
              {processingStatus === 'complete' && 'Analysis complete'}
            </div>
          </div>
        </div>

        {detectedMisconception && (
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="flex items-start gap-2 mb-3">
              <AlertCircle className="h-6 w-6 text-amber-500 flex-shrink-0 mt-0.5" />
              <div>
                <h2 className="font-bold text-lg">Misconception Detected: {detectedMisconception.id}</h2>
                <p className="text-gray-700">{detectedMisconception.description}</p>
              </div>
            </div>
            <div className="bg-blue-50 p-3 rounded-md border border-blue-100">
              <h3 className="font-semibold text-blue-800 flex items-center gap-1 mb-1">
                <Info className="h-4 w-4" /> Feedback
              </h3>
              <p>{feedback}</p>
            </div>
            <div className="mt-4">
              <h3 className="font-semibold text-gray-700 flex items-center gap-1 mb-1">
                <BookOpen className="h-4 w-4" /> Example of this misconception
              </h3>
              <p className="text-sm text-gray-600">{detectedMisconception.example}</p>
            </div>
          </div>
        )}

        {history.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="font-bold text-lg mb-2">Detection History</h2>
            <ul className="divide-y">
              {history.map((item, index) => (
                <li key={index} className="py-2 flex justify-between items-center">
                  <div>
                    <span className="font-medium text-blue-600">{item.id}:</span> {item.description}
                  </div>
                  <span className="text-sm text-gray-500">{item.timestamp}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>

      <footer className="bg-gray-800 text-white p-3 text-center text-sm">
        Powered by Gemini 2.0 Flash &amp; Research by Nancy Otero, Stefania Druga and Andrew Lan
      </footer>
    </div>
  );
};

export default MathMisconceptionDetector;
