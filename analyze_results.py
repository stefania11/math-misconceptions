import os
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Global model instance for reuse
model = None

def load_model():
    global model
    if model is None:
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
    return model

def load_results(experiment_type, model_type="gpt4o"):
    filename = f"{model_type}_experiment_{experiment_type}_final_results.json"
    try:
        print(f"Loading results from {filename}...")
        with open(f'outputs/{filename}', 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results successfully.")
        return results
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def calculate_exact_match_accuracy(results):
    if not results:
        return 0.0
    correct_predictions = sum(1 for result in results
        if (result.get('predicted_misconception', result.get('prediction', '')).strip().lower() ==
            result.get('actual_misconception', result.get('actual', '')).strip().lower()))
    return (correct_predictions / len(results)) * 100

def calculate_semantic_similarity(results):
    if not results:
        return 0.0

    model = load_model()
    similarities = []
    total = len(results)
    print(f"\nCalculating semantic similarity for {total} results...")

    for i, result in enumerate(results, 1):
        if i % 5 == 0:
            print(f"Processing item {i}/{total}")
        try:
            pred = result.get('predicted_misconception', result.get('prediction', ''))
            actual = result.get('actual_misconception', result.get('actual', ''))
            if pred and actual:  # Only process if both values exist
                embeddings = model.encode([pred, actual])
                similarity = 1 - cosine(embeddings[0], embeddings[1])
                similarities.append(similarity)
            else:
                print(f"Warning: Missing prediction or actual value in item {i}")
                similarities.append(0.0)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            similarities.append(0.0)

    avg_similarity = np.mean(similarities) if similarities else 0.0
    print(f"Average semantic similarity: {avg_similarity:.2f}")
    return avg_similarity

def analyze_topic_performance(results):
    if not results:
        return {}

    model = load_model()
    topic_results = {}
    total = len(results)
    print(f"\nProcessing {total} results for topic analysis...")

    for i, result in enumerate(results, 1):
        if i % 5 == 0:
            print(f"Processing item {i}/{total}")

        # Handle different topic key names
        topic = result.get('topic', result.get('test_topic', 'Unknown'))
        if topic not in topic_results:
            topic_results[topic] = {'correct': 0, 'total': 0, 'similarities': []}

        try:
            # Get predictions and actual values using both possible key names
            pred = result.get('predicted_misconception', result.get('prediction', ''))
            actual = result.get('actual_misconception', result.get('actual', ''))

            # Calculate exact match
            if pred.strip().lower() == actual.strip().lower():
                topic_results[topic]['correct'] += 1

            # Calculate semantic similarity
            if pred and actual:  # Only process if both values exist
                embeddings = model.encode([pred, actual])
                similarity = 1 - cosine(embeddings[0], embeddings[1])
                topic_results[topic]['similarities'].append(similarity)
            else:
                print(f"Warning: Missing prediction or actual value in item {i}")
                topic_results[topic]['similarities'].append(0.0)
        except Exception as e:
            print(f"Error processing item {i} for topic {topic}: {e}")
            topic_results[topic]['similarities'].append(0.0)

        topic_results[topic]['total'] += 1

    # Calculate metrics for each topic
    topic_metrics = {}
    print("\nCalculating final metrics for each topic...")
    for topic, stats in topic_results.items():
        if stats['total'] > 0:  # Only calculate metrics if we have results
            topic_metrics[topic] = {
                'accuracy': (stats['correct'] / stats['total']) * 100,
                'avg_similarity': np.mean(stats['similarities']) if stats['similarities'] else 0.0
            }
            print(f"Topic {topic}: Accuracy {topic_metrics[topic]['accuracy']:.2f}%, "
                  f"Similarity {topic_metrics[topic]['avg_similarity']:.2f}")

    return topic_metrics

def main():
    try:
        # Install required package
        print("Installing required packages...")
        os.system('pip install -q sentence-transformers')
        print("Packages installed successfully.")

        # Pre-load the model
        print("\nInitializing model...")
        load_model()

        # Load results for both models
        print("\nLoading experiment results...")
        gpt4o_exp1_results = load_results("1_(cross-topic)")
        gpt4o_exp2_results = load_results("2_(topic-constrained)")
        claude_exp1_results = load_results("1_(cross-topic)", "claude")
        claude_exp2_results = load_results("2_(topic-constrained)", "claude")

        if not all([gpt4o_exp1_results, gpt4o_exp2_results, claude_exp1_results, claude_exp2_results]):
            raise Exception("Failed to load all required result files")

        # Calculate metrics for GPT-4o
        print("\nAnalyzing GPT-4o results...")
        gpt4o_exp1_accuracy = calculate_exact_match_accuracy(gpt4o_exp1_results)
        gpt4o_exp2_accuracy = calculate_exact_match_accuracy(gpt4o_exp2_results)
        gpt4o_exp1_similarity = calculate_semantic_similarity(gpt4o_exp1_results)
        gpt4o_exp2_similarity = calculate_semantic_similarity(gpt4o_exp2_results)
        gpt4o_exp1_topic_metrics = analyze_topic_performance(gpt4o_exp1_results)
        gpt4o_exp2_topic_metrics = analyze_topic_performance(gpt4o_exp2_results)

        # Calculate metrics for Claude
        print("\nAnalyzing Claude results...")
        claude_exp1_accuracy = calculate_exact_match_accuracy(claude_exp1_results)
        claude_exp2_accuracy = calculate_exact_match_accuracy(claude_exp2_results)
        claude_exp1_similarity = calculate_semantic_similarity(claude_exp1_results)
        claude_exp2_similarity = calculate_semantic_similarity(claude_exp2_results)
        claude_exp1_topic_metrics = analyze_topic_performance(claude_exp1_results)
        claude_exp2_topic_metrics = analyze_topic_performance(claude_exp2_results)

        # Create analysis report
        print("\nGenerating analysis report...")
        report = f"""# Math Misconceptions Model Comparison Results

## Experiment Overview
This analysis compares the performance between GPT-4o and Claude Sonnet 3.5 on mathematical misconception identification tasks. Both models were evaluated using the same test cases and experimental conditions.

### Experimental Setup
- **Models Tested**:
  - GPT-4o (OpenAI's latest model)
  - Claude Sonnet 3.5 (baseline comparison)
- **Parameters**:
  - Temperature: 0.2
  - Max tokens: 2000
  - Frequency penalty: 0.0

## Results Analysis

### Experiment 1: Cross-Topic Testing
#### GPT-4o Performance:
- Exact Match Accuracy: {gpt4o_exp1_accuracy:.2f}%
- Semantic Similarity Score: {gpt4o_exp1_similarity:.2f}

#### Claude Sonnet 3.5 Performance:
- Exact Match Accuracy: {claude_exp1_accuracy:.2f}%
- Semantic Similarity Score: {claude_exp1_similarity:.2f}

#### Topic-wise Performance Comparison (Experiment 1):
"""

        # Add topic-wise performance comparison for Experiment 1
        all_topics = set(gpt4o_exp1_topic_metrics.keys()) | set(claude_exp1_topic_metrics.keys())
        for topic in sorted(all_topics):
            gpt4o_metrics = gpt4o_exp1_topic_metrics.get(topic, {'accuracy': 0, 'avg_similarity': 0})
            claude_metrics = claude_exp1_topic_metrics.get(topic, {'accuracy': 0, 'avg_similarity': 0})
            report += f"""
##### {topic}:
- GPT-4o: Accuracy {gpt4o_metrics['accuracy']:.2f}%, Semantic Similarity {gpt4o_metrics['avg_similarity']:.2f}
- Claude Sonnet 3.5: Accuracy {claude_metrics['accuracy']:.2f}%, Semantic Similarity {claude_metrics['avg_similarity']:.2f}"""

        report += f"""

### Experiment 2: Topic-Constrained Testing
#### GPT-4o Performance:
- Exact Match Accuracy: {gpt4o_exp2_accuracy:.2f}%
- Semantic Similarity Score: {gpt4o_exp2_similarity:.2f}

#### Claude Sonnet 3.5 Performance:
- Exact Match Accuracy: {claude_exp2_accuracy:.2f}%
- Semantic Similarity Score: {claude_exp2_similarity:.2f}

#### Topic-wise Performance Comparison (Experiment 2):
"""


        # Add topic-wise performance comparison for Experiment 2
        all_topics = set(gpt4o_exp2_topic_metrics.keys()) | set(claude_exp2_topic_metrics.keys())
        for topic in sorted(all_topics):
            gpt4o_metrics = gpt4o_exp2_topic_metrics.get(topic, {'accuracy': 0, 'avg_similarity': 0})
            claude_metrics = claude_exp2_topic_metrics.get(topic, {'accuracy': 0, 'avg_similarity': 0})
            report += f"""
##### {topic}:
- GPT-4o: Accuracy {gpt4o_metrics['accuracy']:.2f}%, Semantic Similarity {gpt4o_metrics['avg_similarity']:.2f}
- Claude Sonnet 3.5: Accuracy {claude_metrics['accuracy']:.2f}%, Semantic Similarity {claude_metrics['avg_similarity']:.2f}"""

        report += """

## Key Findings
1. Model Performance Comparison:"""

        if gpt4o_exp1_accuracy > claude_exp1_accuracy:
            report += f"\n   - Cross-Topic Testing: GPT-4o outperformed Claude Sonnet 3.5 by {gpt4o_exp1_accuracy - claude_exp1_accuracy:.2f}%"
        else:
            report += f"\n   - Cross-Topic Testing: Claude Sonnet 3.5 outperformed GPT-4o by {claude_exp1_accuracy - gpt4o_exp1_accuracy:.2f}%"

        if gpt4o_exp2_accuracy > claude_exp2_accuracy:
            report += f"\n   - Topic-Constrained Testing: GPT-4o outperformed Claude Sonnet 3.5 by {gpt4o_exp2_accuracy - claude_exp2_accuracy:.2f}%"
        else:
            report += f"\n   - Topic-Constrained Testing: Claude Sonnet 3.5 outperformed GPT-4o by {claude_exp2_accuracy - gpt4o_exp2_accuracy:.2f}%"

        report += f"""

2. Semantic Similarity Analysis:
   - Cross-Topic Testing:
     - GPT-4o average similarity: {gpt4o_exp1_similarity:.2f}
     - Claude Sonnet 3.5 average similarity: {claude_exp1_similarity:.2f}
   - Topic-Constrained Testing:
     - GPT-4o average similarity: {gpt4o_exp2_similarity:.2f}
     - Claude Sonnet 3.5 average similarity: {claude_exp2_similarity:.2f}

## Recommendations
1. Model Selection:"""

        if (gpt4o_exp1_accuracy > claude_exp1_accuracy and gpt4o_exp2_accuracy > claude_exp2_accuracy):
            report += "\n   - GPT-4o demonstrates superior performance across both experiments and should be preferred for mathematical misconception identification tasks."
        elif (gpt4o_exp1_accuracy < claude_exp1_accuracy and gpt4o_exp2_accuracy < claude_exp2_accuracy):
            report += "\n   - Claude Sonnet 3.5 shows consistently better performance across both experiments and should be preferred for mathematical misconception identification tasks."
        else:
            report += "\n   - Both models show strengths in different scenarios. Consider using them in combination or selecting based on specific use cases."

        report += """
2. Future Improvements:
   - Consider fine-tuning models specifically for mathematical misconception identification
   - Explore ensemble approaches combining both models' strengths
   - Investigate cases where semantic similarity is high but exact matches fail
   - Consider expanding the test dataset to cover more mathematical topics and edge cases
"""

        print("Writing analysis report...")
        with open('analysis_results.md', 'w') as f:
            f.write(report)
        print("Analysis complete! Results written to analysis_results.md")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
