[Previous content up to "The experimental outcomes suggest..." remains the same, then add:]

# **Setup and Usage**

## **Prerequisites**
- Python 3.12 or higher
- pip package manager
- OpenAI API key with GPT-4 access

## **Installation**

1. Clone the repository:
```bash
git clone https://github.com/stefania11/math-misconceptions.git
cd math-misconceptions
```

2. Install required packages:
```bash
pip install openai==0.28 pandas tqdm
```

## **Configuration**

### OpenAI API Setup
1. Obtain an OpenAI API key from [OpenAI's platform](https://platform.openai.com)
2. Set up your API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```
   - For Windows: `set OPENAI_API_KEY=your-api-key-here`
   - For permanent setup, add to your shell profile (.bashrc, .zshrc, etc.)

### Dataset Verification
The project requires the MaE dataset, which should be structured as follows:

1. Verify directory structure:
```bash
mkdir -p data/txt_files outputs
```

2. Verify dataset files:
- Ensure `data/data.json` exists and contains the complete dataset
- The dataset should include 55 misconceptions with 4 examples each (220 total)
- Each example should contain:
  - Question
  - Incorrect answer
  - Correct answer
  - Source reference
  - Images (where applicable)

## **Running Tests**

The project includes a test script (`test_setup.py`) to verify your setup:

1. Run the test script:
```bash
python test_setup.py
```

2. Interpreting test results:
- **"OpenAI connection successful!"**: Confirms your API key is valid and working
- **"Data loading successful!"**: Confirms the dataset is present and properly formatted
- If you see both messages, your setup is complete and working correctly

### Troubleshooting Test Output
- If OpenAI connection fails:
  - Verify your API key is set correctly
  - Ensure your API key has GPT-4 access
  - Check your internet connection
- If data loading fails:
  - Verify data.json exists in the data directory
  - Check file permissions
  - Ensure the JSON file is properly formatted

## **Running Experiments**
[Previous content about experimental results continues...]
