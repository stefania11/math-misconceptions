import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import defaultdict

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_core_misconception(text):
    """Extract core misconception from longer text by focusing on key phrases."""
    text = text.lower().strip()
    # Look for common patterns in misconception descriptions
    patterns = ['when students', 'students', 'misconception:', 'the misconception is']
    for pattern in patterns:
        if pattern in text:
            text = text[text.find(pattern):]
            break
    # Truncate explanatory text
    markers = ['. this', '. let', '. for example', '. to solve']
    for marker in markers:
        if marker in text:
            text = text[:text.find(marker)]
    return text.strip()

def analyze_results(model_name, results):
    """Analyze results with topic breakdown and semantic similarity."""
    if not results:
        return {
            'model': model_name,
            'total_accuracy': 0.0,
            'avg_similarity': 0.0,
            'topic_metrics': {},
            'sample_comparisons': []
        }

    model = load_model()
    topic_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'similarities': []})
    sample_comparisons = []

    # Process each result
    for result in results:
        topic = result.get('topic', result.get('test_topic', 'Unknown'))
        pred_full = result.get('predicted_misconception', result.get('prediction', ''))
        actual_full = result.get('actual_misconception', result.get('actual', ''))

        # Extract core misconceptions
        pred = extract_core_misconception(pred_full)
        actual = extract_core_misconception(actual_full)

        # Calculate semantic similarity
        if pred and actual:
            embeddings = model.encode([pred, actual])
            similarity = 1 - cosine(embeddings[0], embeddings[1])
            topic_metrics[topic]['similarities'].append(similarity)

            # Store sample comparisons (first 5 only)
            if len(sample_comparisons) < 5:
                sample_comparisons.append({
                    'topic': topic,
                    'predicted': pred[:200],
                    'actual': actual[:200],
                    'similarity': similarity
                })

        # Update topic metrics
        topic_metrics[topic]['total'] += 1
        if pred.strip() == actual.strip():
            topic_metrics[topic]['correct'] += 1

    # Calculate overall metrics
    total_correct = sum(m['correct'] for m in topic_metrics.values())
    total_samples = sum(m['total'] for m in topic_metrics.values())
    all_similarities = [s for m in topic_metrics.values() for s in m['similarities']]

    # Calculate per-topic metrics
    topic_results = {}
    for topic, metrics in topic_metrics.items():
        if metrics['total'] > 0:
            topic_results[topic] = {
                'accuracy': (metrics['correct'] / metrics['total']) * 100,
                'avg_similarity': np.mean(metrics['similarities']) if metrics['similarities'] else 0.0,
                'total_samples': metrics['total']
            }

    return {
        'model': model_name,
        'total_accuracy': (total_correct / total_samples * 100) if total_samples > 0 else 0.0,
        'avg_similarity': np.mean(all_similarities) if all_similarities else 0.0,
        'topic_metrics': topic_results,
        'sample_comparisons': sample_comparisons
    }

def generate_comparison_report(gpt4o_analysis, claude_analysis):
    """Generate detailed comparison report in markdown format."""
    report = [
        "# Math Misconceptions Model Comparison Results\n",
        "## Overview",
        "This report compares the performance of GPT-4o and Claude Sonnet 3.5 on mathematical misconception identification tasks.\n",
        "## Model Performance\n",
        f"### GPT-4o",
        f"- Exact Match Accuracy: {gpt4o_analysis['total_accuracy']:.2f}%",
        f"- Semantic Similarity Score: {gpt4o_analysis['avg_similarity']:.2f}\n",
        f"### Claude Sonnet 3.5",
        f"- Exact Match Accuracy: {claude_analysis['total_accuracy']:.2f}%",
        f"- Semantic Similarity Score: {claude_analysis['avg_similarity']:.2f}\n",
        "## Topic-wise Performance\n",
        "### Best Performing Topics"
    ]

    # Add topic-wise performance
    for model_analysis in [gpt4o_analysis, claude_analysis]:
        report.extend([
            f"\n#### {model_analysis['model']}"
        ])
        sorted_topics = sorted(
            model_analysis['topic_metrics'].items(),
            key=lambda x: x[1]['avg_similarity'],
            reverse=True
        )
        for topic, metrics in sorted_topics[:3]:
            report.extend([
                f"- {topic}:",
                f"  - Accuracy: {metrics['accuracy']:.2f}%",
                f"  - Semantic Similarity: {metrics['avg_similarity']:.2f}",
                f"  - Samples: {metrics['total_samples']}"
            ])

    # Add sample comparisons
    report.extend([
        "\n## Sample Comparisons",
        "\nBelow are sample comparisons showing how each model interpreted misconceptions:"
    ])

    for model_analysis in [gpt4o_analysis, claude_analysis]:
        report.extend([
            f"\n### {model_analysis['model']} Samples"
        ])
        for i, sample in enumerate(model_analysis['sample_comparisons'][:3], 1):
            report.extend([
                f"\n#### Sample {i} ({sample['topic']})",
                f"- Predicted: {sample['predicted']}",
                f"- Actual: {sample['actual']}",
                f"- Similarity Score: {sample['similarity']:.2f}"
            ])

    report.extend([
        "\n## Methodology",
        "- Models compared: GPT-4o and Claude Sonnet 3.5",
        "- Evaluation metrics: Exact match accuracy and semantic similarity using SentenceTransformer",
        "- Dataset: Cross-topic testing with mathematical misconceptions",
        "- Analysis includes core misconception extraction and semantic similarity scoring"
    ])

    return "\n".join(report)

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load and analyze results
    with open('outputs/gpt4o_experiment_1_(cross-topic)_final_results.json', 'r') as f:
        gpt4o_results = json.load(f)
    with open('outputs/claude_experiment_1_(cross-topic)_final_results.json', 'r') as f:
        claude_results = json.load(f)

    print("Analyzing GPT-4o results...")
    gpt4o_analysis = analyze_results('GPT-4o', gpt4o_results)
    print("Analyzing Claude results...")
    claude_analysis = analyze_results('Claude Sonnet', claude_results)

    # Generate and save report
    print("Generating comparison report...")
    report = generate_comparison_report(gpt4o_analysis, claude_analysis)
    with open('analysis_results.md', 'w') as f:
        f.write(report)
    print('Analysis complete. Report generated in analysis_results.md')
