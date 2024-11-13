# Math Misconceptions Model Comparison Results

## Overview
This report compares the performance of GPT-4o and Claude Sonnet 3.5 on mathematical misconception identification tasks.

## Model Performance

### GPT-4o
- Exact Match Accuracy: 0.00%
- Semantic Similarity Score: 0.38

### Claude Sonnet 3.5
- Exact Match Accuracy: 0.00%
- Semantic Similarity Score: 0.49

## Topic-wise Performance

### Best Performing Topics

#### GPT-4o
- Number Operations:
  - Accuracy: 0.00%
  - Semantic Similarity: 0.52
  - Samples: 32
- Algebraic representations:
  - Accuracy: 0.00%
  - Semantic Similarity: 0.42
  - Samples: 2
- Ratios and proportional reasoning:
  - Accuracy: 0.00%
  - Semantic Similarity: 0.39
  - Samples: 17

#### Claude Sonnet
- Number Operations:
  - Accuracy: 0.00%
  - Semantic Similarity: 0.57
  - Samples: 24
- Properties of number and operations:
  - Accuracy: 0.00%
  - Semantic Similarity: 0.54
  - Samples: 8
- Ratios and proportional reasoning:
  - Accuracy: 0.00%
  - Semantic Similarity: 0.50
  - Samples: 14

## Sample Comparisons

Below are sample comparisons showing how each model interpreted misconceptions:

### GPT-4o Samples

#### Sample 1 (Number Operations)
- Predicted: the incorrect answer provided contains a few misconceptions in the process of solving the equation
- Actual: when students subtract mixed numbers incorrectly, avoiding regrouping and just subtracting the smaller from the larger number
- Similarity Score: 0.18

#### Sample 2 (Patterns, relationships, and functions)
- Predicted: to determine which group is growing faster at the age of 14, we need to look at the steepness of the lines on the graph at that age. the steeper the line, the faster the growth rate.

at age 14, the d
- Actual: when students struggle to grasp the concept that a linear function represents a consistent rate of change
- Similarity Score: 0.31

#### Sample 3 (Patterns, relationships, and functions)
- Predicted: the graph shows a line with a positive slope, not a negative one
- Actual: when students misinterpret slope signs in equations versus their upward or downward trends in graphs.
- Similarity Score: 0.54

### Claude Sonnet Samples

#### Sample 1 (Number Operations)
- Predicted: failure to understand fraction operations and applying whole number operations to fractions.
- Actual: when students wrongly divide fractions by splitting numerators and denominators into separate divisions, ignoring remainders
- Similarity Score: 0.70

#### Sample 2 (Algebraic representations)
- Predicted: misunderstanding the order of coordinates (x, y) on the cartesian plane.
- Actual: when students struggle with plotting points, reversing the x- and y-coordinates
- Similarity Score: 0.42

#### Sample 3 (Number sense)
- Predicted: lack of understanding of common factors and prime factorization to simplify fractions.
- Actual: students misunderstand proportional relationships, not realizing parts must be equal in size.
- Similarity Score: 0.42

## Methodology
- Models compared: GPT-4o and Claude Sonnet 3.5
- Evaluation metrics: Exact match accuracy and semantic similarity using SentenceTransformer
- Dataset: Cross-topic testing with mathematical misconceptions
- Analysis includes core misconception extraction and semantic similarity scoring