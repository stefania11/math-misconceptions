import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize the Gemini API with your API key
const API_KEY = process.env.NEXT_PUBLIC_GEMINI_API || '';
const genAI = new GoogleGenerativeAI(API_KEY);

// Specify the model (Gemini Flash 2.0)
const modelName = 'gemini-2.0-flash';

export interface MathMisconception {
  id: string;
  description: string;
  example: string;
  feedback: string;
}

export const analyzeMathProblem = async (imageBase64: string): Promise<MathMisconception | null> => {
  try {
    // Remove the data URL prefix if present
    const base64Data = imageBase64.replace(/^data:image\/(png|jpeg|jpg);base64,/, '');
    
    // Get the model
    const model = genAI.getGenerativeModel({ model: modelName });
    
    // Prepare the prompt
    const prompt = `
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
        "id": "MaEXX",
        "description": "Brief description of the misconception",
        "example": "Example of this type of misconception",
        "feedback": "Helpful feedback for the student"
      }
      
      If you cannot identify a clear misconception, respond with null.
    `;
    
    // Create the image part for the request
    const imageParts = [
      {
        inlineData: {
          data: base64Data,
          mimeType: 'image/jpeg'
        }
      }
    ];
    
    // Generate content with the image
    const result = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: prompt }, ...imageParts] }],
    });
    
    const response = result.response;
    const text = response.text();
    
    // Parse the JSON response
    try {
      // Extract JSON from the response if needed
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const jsonStr = jsonMatch[0];
        const misconception = JSON.parse(jsonStr) as MathMisconception;
        return misconception;
      }
      return null;
    } catch (parseError) {
      console.error('Error parsing Gemini response:', parseError);
      return null;
    }
  } catch (error) {
    console.error('Error calling Gemini API:', error);
    return null;
  }
};
