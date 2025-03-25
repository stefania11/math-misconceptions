# Math Misconception Detector

An interactive prototype for middle schoolers to learn algebra using multimodal AI. This application uses the Gemini Flash 2.0 model to detect and provide feedback on common algebra misconceptions.

## Features

- Camera-based capture of math work
- Real-time analysis using Gemini Flash 2.0 AI
- Detection of common algebra misconceptions
- Personalized feedback for students
- History tracking of detected misconceptions

## Technology Stack

- Next.js
- React
- TypeScript
- Tailwind CSS
- Google Generative AI (Gemini Flash 2.0)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env.local` file with your Gemini API key:
   ```
   NEXT_PUBLIC_GEMINI_API=your_api_key_here
   ```
4. Run the development server:
   ```bash
   npm run dev
   ```

## Usage

1. Allow camera access when prompted
2. Point the camera at a student's math work
3. Click the "Capture" button
4. The AI will analyze the work and detect any misconceptions
5. Review the feedback and example provided

## Misconception Database

The application includes a sample database of common algebra misconceptions based on the MaE (Math Misconceptions and Errors) dataset, including:

- Simplifying just one term in a fraction (MaE07)
- Subtracting mixed numbers incorrectly (MaE11)
- Struggling with independent/dependent variables (MaE44)
- Difficulty with variable meanings (MaE49)
- Confusion with operation symbols (MaE53)

## License

MIT
