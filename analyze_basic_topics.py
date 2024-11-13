import json
from collections import defaultdict

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_topic_performance(results, model_type):
    topic_metrics = defaultdict(lambda: {'count': 0, 'exact_matches': 0})

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

        exact_match = 1 if actual.lower() == predicted.lower() else 0

        topic_metrics[topic]['count'] += 1
        topic_metrics[topic]['exact_matches'] += exact_match

        # Store an example of prediction for analysis
        if 'examples' not in topic_metrics[topic]:
            topic_metrics[topic]['examples'] = []
        if len(topic_metrics[topic]['examples']) < 2:  # Store up to 2 examples per topic
            topic_metrics[topic]['examples'].append({
                'actual': actual,
                'predicted': predicted,
                'exact_match': exact_match == 1
            })

    # Calculate success rates
    topic_performance = {}
    for topic, metrics in topic_metrics.items():
        if metrics['count'] > 0:
            success_rate = (metrics['exact_matches'] / metrics['count']) * 100
            topic_performance[topic] = {
                'count': metrics['count'],
                'exact_match_rate': success_rate,
                'examples': metrics['examples']
            }

    return topic_performance

def main():
    # Load results
    print("Loading experiment results...")
    gpt4o_exp1 = load_results('outputs/gpt4o_experiment_1_(cross-topic)_final_results.json')
    claude_exp1 = load_results('outputs/claude_experiment_1_(cross-topic)_final_results.json')

    print("\nAnalyzing GPT-4o performance...")
    gpt4o_performance = analyze_topic_performance(gpt4o_exp1, 'gpt4o')

    print("Analyzing Claude performance...")
    claude_performance = analyze_topic_performance(claude_exp1, 'claude')

    # Sort topics by success rate (ascending)
    gpt4o_sorted = sorted(gpt4o_performance.items(), key=lambda x: x[1]['exact_match_rate'])
    claude_sorted = sorted(claude_performance.items(), key=lambda x: x[1]['exact_match_rate'])

    # Generate report
    report = "# Topic Performance Analysis (Basic Metrics)\n\n"

    report += "## GPT-4o Worst Performing Topics\n"
    for topic, metrics in gpt4o_sorted[:3]:
        report += f"\n### {topic}\n"
        report += f"- Exact Match Rate: {metrics['exact_match_rate']:.1f}%\n"
        report += f"- Sample Size: {metrics['count']}\n"
        report += "\nExample Predictions:\n"
        for ex in metrics['examples']:
            report += f"\nActual: {ex['actual']}\nPredicted: {ex['predicted']}\nExact Match: {ex['exact_match']}\n"
        report += "---\n"

    report += "\n## Claude Sonnet Worst Performing Topics\n"
    for topic, metrics in claude_sorted[:3]:
        report += f"\n### {topic}\n"
        report += f"- Exact Match Rate: {metrics['exact_match_rate']:.1f}%\n"
        report += f"- Sample Size: {metrics['count']}\n"
        report += "\nExample Predictions:\n"
        for ex in metrics['examples']:
            report += f"\nActual: {ex['actual']}\nPredicted: {ex['predicted']}\nExact Match: {ex['exact_match']}\n"
        report += "---\n"

    print("\nWriting report...")
    with open('outputs/topic_performance_basic.md', 'w') as f:
        f.write(report)
    print("Analysis complete! Report saved to outputs/topic_performance_basic.md")

if __name__ == "__main__":
    main()
