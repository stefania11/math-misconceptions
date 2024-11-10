import json
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_semantic_similarity(model, text1, text2):
    # Encode the texts and calculate cosine similarity
    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return similarity

def analyze_results(experiment_name, results, model, similarity_threshold=0.7):
    total = len(results)
    semantic_matches = 0
    similarities = []
    examples = []

    # Group by topics for analysis
    topic_accuracies = defaultdict(lambda: {'matches': 0, 'total': 0})

    for result in results:
        similarity = calculate_semantic_similarity(
            model,
            result['prediction'].lower(),
            result['actual'].lower()
        )
        similarities.append(similarity)

        if similarity >= similarity_threshold:
            semantic_matches += 1

        # Track topic-specific performance
        test_topic = result['test_topic']
        topic_accuracies[test_topic]['total'] += 1
        if similarity >= similarity_threshold:
            topic_accuracies[test_topic]['matches'] += 1

        # Store example for analysis
        if len(examples) < 3 or similarity > 0.8:  # Store first 3 and high-similarity examples
            examples.append({
                'prediction': result['prediction'],
                'actual': result['actual'],
                'similarity': similarity,
                'training_topic': result['training_topic'],
                'test_topic': result['test_topic']
            })

    # Calculate metrics
    semantic_accuracy = semantic_matches / total
    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)

    # Calculate topic-specific accuracies
    topic_specific_accuracies = {
        topic: {'accuracy': data['matches'] / data['total'], 'count': data['total']}
        for topic, data in topic_accuracies.items()
    }

    return {
        'experiment_name': experiment_name,
        'semantic_accuracy': semantic_accuracy,
        'average_similarity': avg_similarity,
        'median_similarity': median_similarity,
        'total_examples': total,
        'semantic_matches': semantic_matches,
        'topic_accuracies': topic_specific_accuracies,
        'examples': examples
    }

def main():
    # Load the model
    print("Loading BERT model for semantic similarity analysis...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Analyze both experiments
    experiments = [
        ('Cross-Topic Testing', 'outputs/claude_experiment_1_(cross-topic)_final_results.json'),
        ('Topic-Constrained Testing', 'outputs/claude_experiment_2_(topic-constrained)_final_results.json')
    ]

    print("\nAnalyzing results using semantic similarity...\n")

    for exp_name, filename in experiments:
        results = load_results(filename)
        analysis = analyze_results(exp_name, results, model)

        print(f"\n=== {exp_name} Analysis ===")
        print(f"Semantic Accuracy (threshold 0.7): {analysis['semantic_accuracy']:.2%}")
        print(f"Average Similarity Score: {analysis['average_similarity']:.3f}")
        print(f"Median Similarity Score: {analysis['median_similarity']:.3f}")
        print(f"Total Examples: {analysis['total_examples']}")
        print(f"Semantic Matches: {analysis['semantic_matches']}")

        print("\nTop Topic Performances:")
        sorted_topics = sorted(
            analysis['topic_accuracies'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )[:3]
        for topic, stats in sorted_topics:
            print(f"- {topic}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print("\nExample Predictions:")
        for i, example in enumerate(analysis['examples'][:3], 1):
            print(f"\nExample {i}:")
            print(f"Prediction: {example['prediction']}")
            print(f"Actual: {example['actual']}")
            print(f"Similarity Score: {example['similarity']:.3f}")
            print(f"Topics: {example['training_topic']} -> {example['test_topic']}")

if __name__ == "__main__":
    main()
