from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize the Gemini API with your API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY environment variable not set')

genai.configure(api_key=api_key)

# Specify the model (Gemini Flash 2.0)
model_name = 'gemini-2.0-flash'

# Create FastAPI app
app = FastAPI(
    title='Math Misconception API',
    description='API for detecting math misconceptions using Gemini Flash 2.0',
    version='1.0.0'
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://math-misconception-app-1sxqnxsl.devinapps.com', 'http://localhost:8000', 'http://localhost:8001'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Sample misconception database
misconception_database = [
    {
        'id': 'MaE07',
        'description': 'Simplifying just one term in a fraction',
        'example': 'Simplifying only the numerator or denominator in a fraction',
        'feedback': 'Remember to apply the same operation to both the numerator and denominator when simplifying fractions.'
    },
    {
        'id': 'MaE11',
        'description': 'Subtracting mixed numbers incorrectly',
        'example': 'Avoiding regrouping and just subtracting smaller from larger',
        'feedback': 'When subtracting mixed numbers, you may need to regroup (borrow) when the fraction part of the first number is smaller than the fraction part of the second number.'
    },
    {
        'id': 'MaE44',
        'description': 'Struggling with independent/dependent variables',
        'example': 'Difficulty identifying which variable depends on the other',
        'feedback': 'The dependent variable (output) changes based on the independent variable (input). Think about cause and effect - which variable controls the other?'
    },
    {
        'id': 'MaE49',
        'description': 'Difficulty with variable meanings',
        'example': 'Struggling to comprehend various meanings and applications of variables',
        'feedback': 'Variables can represent unknown values, changing quantities, or relationships between values. In this problem, try to clearly identify what the variable represents.'
    },
    {
        'id': 'MaE53',
        'description': 'Confusion with operation symbols',
        'example': 'Mixing up operation symbols and their meanings in algebra',
        'feedback': 'Be careful with operation symbols. Remember that each symbol (+, -, ×, ÷, =) has a specific meaning in the context of an equation.'
    }
]

@app.get('/')
async def root():
    return {'message': 'Math Misconception API is running'}

@app.get('/api/misconceptions')
async def get_misconceptions():
    return misconception_database

@app.post('/api/analyze')
async def analyze_math_work(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    # Check if we have either a file or base64 image
    if not file and not image_base64:
        raise HTTPException(status_code=400, detail='Either file or image_base64 must be provided')
    
    try:
        # Process the image
        if file:
            # Read the uploaded file
            contents = await file.read()
            # Convert to base64
            image_base64 = base64.b64encode(contents).decode('utf-8')
        else:
            # Remove data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
        
        # Get the model
        model = genai.GenerativeModel(model_name)
        
        # Prepare the prompt
        prompt = '''
        You are an expert math tutor for middle school students. 
        Analyze this image of a student's math work and identify any algebra misconceptions.
        Focus on common middle school algebra misconceptions like:
        - Simplifying just one term in a fraction
        - Subtracting mixed numbers incorrectly
        - Struggling with independent/dependent variables
        - Difficulty with variable meanings
        - Confusion with operation symbols
        
        If you identify a misconception, provide:
        1. A misconception ID (e.g., MaE07, MaE11, MaE44, etc.)
        2. A brief description of the misconception
        3. An example of this type of misconception
        4. Helpful, encouraging feedback for the student
        
        Format your response as a JSON object with these fields:
        {
          'id': 'MaEXX',
          'description': 'Brief description of the misconception',
          'example': 'Example of this type of misconception',
          'feedback': 'Helpful feedback for the student'
        }
        
        If you cannot identify a clear misconception, respond with null.
        '''
        
        # Create image data for the request
        image_data = {'mime_type': 'image/jpeg', 'data': image_base64}
        
        # Generate content with the image
        response = model.generate_content([prompt, image_data])
        
        # Parse the response
        text = response.text
        
        # Extract JSON from the response if needed
        try:
            # Look for JSON pattern in the response
            json_match = None
            for line in text.split('\n'):
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    json_match = line.strip()
                    break
            
            if not json_match:
                # Try to find JSON with regex
                import re
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, text)
                if matches:
                    json_match = matches[0]
            
            if json_match:
                misconception = json.loads(json_match)
                return misconception
            elif 'null' in text.lower():
                return None
            else:
                # Fallback to a default response if JSON parsing fails
                return {
                    'id': 'MaE00',
                    'description': 'Unable to identify specific misconception',
                    'example': 'The image may not contain clear algebra work or misconceptions',
                    'feedback': 'Please ensure the image clearly shows your math work. Try taking a clearer photo or writing more legibly.'
                }
        except Exception as e:
            print(f'Error parsing Gemini response: {e}')
            # Return a fallback response
            return {
                'id': 'MaE00',
                'description': 'Error processing response',
                'example': 'The AI had trouble analyzing this specific image',
                'feedback': 'Please try again with a clearer image of your math work.'
            }
            
    except Exception as e:
        print(f'Error calling Gemini API: {e}')
        raise HTTPException(status_code=500, detail=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)

