# **MaE: Math Misconceptions and Errors Dataset**

<!-- Center and enlarge the main links -->
<p align="center" style="font-size: 1.5em;">
    <a href="https://huggingface.co/datasets/nanote/algebra_misconceptions">HuggingFace</a>

This dataset supports the research described in the paper [A Benchmark for Math Misconceptions: Bridging Gaps in Middle School Algebra with AI-Supported Instruction](https://arxiv.org/abs/your-link) by Nancy Otero, Stefania Druga, and Andrew Lan.

### **Overview**
The **MaE (Math Misconceptions and Errors)** dataset is a collection of 220 diagnostic examples designed by math learning researchers that represent 55 common algebra misconceptions among middle school students. It aims to provide insights into student errors and misconceptions in algebra, supporting the development of AI-enhanced educational tools that can improve math instruction and learning outcomes.

## **Setup and Usage**

### Prerequisites
- Python 3.12 or higher
- pip package manager
- OpenAI API key with GPT-4 access

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/stefania11/math-misconceptions.git
   cd math-misconceptions
   ```

2. Install required packages:
   ```bash
   pip install openai==0.28 pandas tqdm
   ```

### Configuration

#### OpenAI API Setup
1. Obtain an OpenAI API key from [OpenAI's platform](https://platform.openai.com)
2. Set up your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   - For Windows: `set OPENAI_API_KEY=your-api-key-here`
   - For permanent setup, add to your shell profile (.bashrc, .zshrc, etc.)

#### Dataset Verification
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

### Running Tests

The project includes a test script (`test_setup.py`) to verify your setup:

1. Run the test script:
   ```bash
   python test_setup.py
   ```

2. Interpreting test results:
   - **"OpenAI connection successful!"**: Confirms your API key is valid and working
   - **"Data loading successful!"**: Confirms the dataset is present and properly formatted
   - If you see both messages, your setup is complete and working correctly

#### Troubleshooting
- If OpenAI connection fails:
  - Verify your API key is set correctly
  - Ensure your API key has GPT-4 access
  - Check your internet connection
- If data loading fails:
  - Verify data.json exists in the data directory
  - Check file permissions
  - Ensure the JSON file is properly formatted

## **Dataset Details**

* Total Misconceptions: 55
* Total Examples: 220
* Topics Covered:

1. **Number sense** (MaE01-MaE05)
   - Understanding numbers and their relationships
2. **Number operations** (MaE06-MaE22)
   - Integer subtraction
   - Fractions and decimal operations
   - Order of operations
3. **Ratios and proportional reasoning** (MaE23-MaE28)
   - Ratio concepts
   - Proportional thinking
   - Problem-solving with ratios
4. **Properties of numbers and operations** (MaE31-MaE34)
   - Commutative, associative, and distributive properties
   - Algebraic manipulations
   - Order of operations
5. **Patterns, relationships, and functions** (MaE35-MaE42)
   - Pattern analysis and generalization
   - Tables, graphs, and symbolic rules
   - Function relationships
6. **Algebraic representations** (MaE43-MaE44)
   - Symbolic expressions and graphs
   - Multiple representations
   - Linear equations
7. **Variables, expressions, and operations** (MaE45-MaE48)
   - Expression structure
   - Polynomial arithmetic
   - Equation creation and reasoning
8. **Equations and inequalities** (MaE49-MaE55)
   - Linear equations and inequalities
   - Proportional relationships
   - Function modeling
