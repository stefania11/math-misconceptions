import json
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_similarity(text1, text2, model):
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def analyze_topic_performance(results, model_type, sentence_model):
    topic_metrics = defaultdict(lambda: {'count': 0, 'similarities': [], 'exact_matches': 0})

    for result in results:
        if model_type == 'gpt4o':
            topic = result.get('topic', 'unknown')
            actual = result.get('actual_misconception', '')
            predicted = result.get('predicted_misconception', '')
        else:  # claude
            topic = result.get('test_topic', 'unknown')
            actual = result.get('actual', '')
            predicted = result.get('prediction', '')

        if not actual or not predicted:
            continue

        similarity = calculate_similarity(actual, predicted, sentence_model)
        exact_match = 1 if actual.lower() == predicted.lower() else 0

        topic_metrics[topic]['count'] += 1
        topic_metrics[topic]['similarities'].append(similarity)
        topic_metrics[topic]['exact_matches'] += exact_match

    # Calculate aggregated metrics per topic
    topic_performance = {}
    for topic, metrics in topic_metrics.items():
        if metrics['count'] > 0:
            avg_similarity = np.mean(metrics['similarities'])
            exact_match_rate = metrics['exact_matches'] / metrics['count'] * 100
            above_threshold = sum(1 for s in metrics['similarities'] if s >= 0.7)
            semantic_success_rate = (above_threshold / metrics['count']) * 100

            topic_performance[topic] = {
                'count': metrics['count'],
                'avg_similarity': avg_similarity,
                'exact_match_rate': exact_match_rate,
                'semantic_success_rate': semantic_success_rate
            }

    return topic_performance

def main():
    # Load results
    gpt4o_exp1 = load_results('outputs/gpt4o_experiment_1_(cross-topic)_final_results.json')
    claude_exp1 = load_results('outputs/claude_experiment_1_(cross-topic)_final_results.json')

    # Initialize sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Analyze performance by topic
    gpt4o_performance = analyze_topic_performance(gpt4o_exp1, 'gpt4o', model)
    claude_performance = analyze_topic_performance(claude_exp1, 'claude', model)

    # Sort topics by average similarity (ascending)
    gpt4o_sorted = sorted(gpt4o_performance.items(), key=lambda x: x[1]['avg_similarity'])
    claude_sorted = sorted(claude_performance.items(), key=lambda x: x[1]['avg_similarity'])

    # Generate report
    report = "# Topic Performance Analysis\n\n"

    report += "## GPT-4o Worst Performing Topics\n"
    for topic, metrics in gpt4o_sorted[:3]:
        report += f"\n### {topic}\n"
        report += f"- Average Similarity: {metrics['avg_similarity']:.3f}\n"
        report += f"- Semantic Success Rate: {metrics['semantic_success_rate']:.1f}%\n"
        report += f"- Exact Match Rate: {metrics['exact_match_rate']:.1f}%\n"
        report += f"- Sample Size: {metrics['count']}\n"

    report += "\n## Claude Sonnet Worst Performing Topics\n"
    for topic, metrics in claude_sorted[:3]:
        report += f"\n### {topic}\n"
        report += f"- Average Similarity: {metrics['avg_similarity']:.3f}\n"
        report += f"- Semantic Success Rate: {metrics['semantic_success_rate']:.1f}%\n"
        report += f"- Exact Match Rate: {metrics['exact_match_rate']:.1f}%\n"
        report += f"- Sample Size: {metrics['count']}\n"

    with open('outputs/topic_performance_analysis.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
