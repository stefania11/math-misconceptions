import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_semantic_similarity(model, text1, text2):
    # Encode the texts into embeddings
    embeddings = model.encode([text1, text2])
    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def get_misconception_pair(result, model_type):
    if model_type == 'gpt4o':
        return result.get('actual_misconception', ''), result.get('predicted_misconception', '')
    elif model_type == 'claude':
        return result.get('actual', ''), result.get('prediction', '')
    return '', ''

def analyze_results(gpt4o_results, claude_results, similarity_threshold=0.7):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    results = {
        'exact_match': {'gpt4o': 0, 'claude': 0},
        'semantic_match': {'gpt4o': 0, 'claude': 0},
        'total_examples': {'gpt4o': len(gpt4o_results), 'claude': len(claude_results)},
        'similarities_gpt4o': [],
        'similarities_claude': []
    }

    # Analyze GPT-4o results
    for result in gpt4o_results:
        actual, predicted = get_misconception_pair(result, 'gpt4o')

        # Skip empty results
        if not actual or not predicted:
            continue

        # Check exact match
        if actual.lower() == predicted.lower():
            results['exact_match']['gpt4o'] += 1

        # Calculate semantic similarity
        similarity = calculate_semantic_similarity(model, actual, predicted)
        results['similarities_gpt4o'].append(similarity)
        if similarity >= similarity_threshold:
            results['semantic_match']['gpt4o'] += 1

    # Analyze Claude results
    for result in claude_results:
        actual, predicted = get_misconception_pair(result, 'claude')

        # Skip empty results
        if not actual or not predicted:
            continue

        # Check exact match
        if actual.lower() == predicted.lower():
            results['exact_match']['claude'] += 1

        # Calculate semantic similarity
        similarity = calculate_semantic_similarity(model, actual, predicted)
        results['similarities_claude'].append(similarity)
        if similarity >= similarity_threshold:
            results['semantic_match']['claude'] += 1

    return results

def generate_report(exp1_results, exp2_results):
    report = """# Model Comparison Report: GPT-4o vs Claude Sonnet

## Experiment 1 (Cross-Topic Testing)

### GPT-4o Results
- Total Examples: {gpt4o_total_1}
- Exact Match Accuracy: {gpt4o_exact_1:.2f}%
- Semantic Similarity Accuracy (threshold 0.7): {gpt4o_semantic_1:.2f}%
- Average Semantic Similarity: {gpt4o_avg_sim_1:.2f}

### Claude Sonnet Results
- Total Examples: {claude_total_1}
- Exact Match Accuracy: {claude_exact_1:.2f}%
- Semantic Similarity Accuracy (threshold 0.7): {claude_semantic_1:.2f}%
- Average Semantic Similarity: {claude_avg_sim_1:.2f}

## Experiment 2 (Topic-Constrained Testing)

### GPT-4o Results
- Total Examples: {gpt4o_total_2}
- Exact Match Accuracy: {gpt4o_exact_2:.2f}%
- Semantic Similarity Accuracy (threshold 0.7): {gpt4o_semantic_2:.2f}%
- Average Semantic Similarity: {gpt4o_avg_sim_2:.2f}

### Claude Sonnet Results
- Total Examples: {claude_total_2}
- Exact Match Accuracy: {claude_exact_2:.2f}%
- Semantic Similarity Accuracy (threshold 0.7): {claude_semantic_2:.2f}%
- Average Semantic Similarity: {claude_avg_sim_2:.2f}

## Analysis Summary
{summary}
""".format(
        gpt4o_total_1=exp1_results['total_examples']['gpt4o'],
        gpt4o_exact_1=exp1_results['exact_match']['gpt4o'] / exp1_results['total_examples']['gpt4o'] * 100 if exp1_results['total_examples']['gpt4o'] > 0 else 0,
        gpt4o_semantic_1=exp1_results['semantic_match']['gpt4o'] / exp1_results['total_examples']['gpt4o'] * 100 if exp1_results['total_examples']['gpt4o'] > 0 else 0,
        gpt4o_avg_sim_1=np.mean(exp1_results['similarities_gpt4o']) if exp1_results['similarities_gpt4o'] else 0,
        claude_total_1=exp1_results['total_examples']['claude'],
        claude_exact_1=exp1_results['exact_match']['claude'] / exp1_results['total_examples']['claude'] * 100 if exp1_results['total_examples']['claude'] > 0 else 0,
        claude_semantic_1=exp1_results['semantic_match']['claude'] / exp1_results['total_examples']['claude'] * 100 if exp1_results['total_examples']['claude'] > 0 else 0,
        claude_avg_sim_1=np.mean(exp1_results['similarities_claude']) if exp1_results['similarities_claude'] else 0,
        gpt4o_total_2=exp2_results['total_examples']['gpt4o'],
        gpt4o_exact_2=exp2_results['exact_match']['gpt4o'] / exp2_results['total_examples']['gpt4o'] * 100 if exp2_results['total_examples']['gpt4o'] > 0 else 0,
        gpt4o_semantic_2=exp2_results['semantic_match']['gpt4o'] / exp2_results['total_examples']['gpt4o'] * 100 if exp2_results['total_examples']['gpt4o'] > 0 else 0,
        gpt4o_avg_sim_2=np.mean(exp2_results['similarities_gpt4o']) if exp2_results['similarities_gpt4o'] else 0,
        claude_total_2=exp2_results['total_examples']['claude'],
        claude_exact_2=exp2_results['exact_match']['claude'] / exp2_results['total_examples']['claude'] * 100 if exp2_results['total_examples']['claude'] > 0 else 0,
        claude_semantic_2=exp2_results['semantic_match']['claude'] / exp2_results['total_examples']['claude'] * 100 if exp2_results['total_examples']['claude'] > 0 else 0,
        claude_avg_sim_2=np.mean(exp2_results['similarities_claude']) if exp2_results['similarities_claude'] else 0,
        summary="""
The comparison reveals several key insights:
1. Cross-Topic Testing shows differences in model performance across diverse mathematical topics
2. Topic-Constrained Testing demonstrates each model's ability to leverage domain-specific context
3. Semantic similarity analysis provides a more nuanced view of model understanding beyond exact matches
"""
    )

    return report

if __name__ == "__main__":
    try:
        # Load results
        print("Loading experiment results...")
        gpt4o_exp1 = load_results('outputs/gpt4o_experiment_1_(cross-topic)_final_results.json')
        gpt4o_exp2 = load_results('outputs/gpt4o_experiment_2_(topic-constrained)_final_results.json')
        claude_exp1 = load_results('outputs/claude_experiment_1_(cross-topic)_final_results.json')
        claude_exp2 = load_results('outputs/claude_experiment_2_(topic-constrained)_final_results.json')

        # Analyze results
        print("Analyzing Experiment 1 results...")
        exp1_results = analyze_results(gpt4o_exp1, claude_exp1)

        print("Analyzing Experiment 2 results...")
        exp2_results = analyze_results(gpt4o_exp2, claude_exp2)

        # Generate and save report
        print("Generating comparison report...")
        report = generate_report(exp1_results, exp2_results)

        with open('outputs/model_comparison_report.md', 'w') as f:
            f.write(report)

        print("Report saved to outputs/model_comparison_report.md")
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the result files: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
