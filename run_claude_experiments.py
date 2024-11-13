import os
import anthropic
import json
import random
from tqdm import tqdm
from collections import defaultdict

def load_data():
    """Load and organize the dataset from data.json"""
    with open('data/data.json', 'r') as f:
        data = json.load(f)

    # Organize data by topic
    topics = defaultdict(list)
    for example in data:
        topics[example['Topic']].append(example)

    return data, topics

def format_prompt(training_example, test_example):
    """Format the prompt for Claude in a clear, structured way"""
    return f"""Training Example:
Question: {training_example['Question']}
Incorrect Answer: {training_example['Incorrect Answer']}
Correct Answer: {training_example['Correct Answer']}
Misconception: {training_example['Misconception']}

Test Example:
Question: {test_example['Question']}
Incorrect Answer: {test_example['Incorrect Answer']}
Correct Answer: {test_example['Correct Answer']}

Based on the training example above, what specific misconception is demonstrated in the test example?
Respond with just the misconception name/description."""

def run_experiment(client, training_example, test_example):
    """Run a single experiment with Claude"""
    prompt = format_prompt(training_example, test_example)

    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2000,
        temperature=0.2,
        system="You are an expert in identifying mathematical misconceptions in student work. Analyze the given examples and identify the specific misconception demonstrated.",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    # Extract the text content from the message response
    prediction = message.content[0].text if isinstance(message.content, list) else message.content

    return {
        'prediction': prediction,
        'actual': test_example['Misconception'],
        'training_topic': training_example['Topic'],
        'test_topic': test_example['Topic']
    }

def main():
    print("Loading dataset...")
    all_examples, topics_dict = load_data()

    # Initialize Anthropic client
    client = anthropic.Anthropic(
        api_key=os.environ.get('anthropic_key')
    )

    # Prepare experiments
    experiments = [
        ("Experiment 1 (Cross-Topic)", False),
        ("Experiment 2 (Topic-Constrained)", True)
    ]

    for exp_name, is_topic_constrained in experiments:
        print(f"\nRunning {exp_name}")
        results = []

        for i in tqdm(range(100)):  # 100 iterations as in original
            if is_topic_constrained:
                # Topic-constrained selection
                topic = random.choice(list(topics_dict.keys()))
                topic_examples = topics_dict[topic]
                training = random.choice(topic_examples)
                test = random.choice([x for x in topic_examples if x != training])
            else:
                # Cross-topic selection
                training = random.choice(all_examples)
                test = random.choice([x for x in all_examples
                                    if x['Topic'] != training['Topic']])

            # Run experiment
            result = run_experiment(client, training, test)
            results.append(result)

            # Save intermediate results every 10 iterations
            if (i + 1) % 10 == 0:
                os.makedirs('outputs', exist_ok=True)
                with open(f'outputs/claude_{exp_name.lower().replace(" ", "_")}_results_{i+1}.json', 'w') as f:
                    json.dump(results, f, indent=2)

        # Save final results
        with open(f'outputs/claude_{exp_name.lower().replace(" ", "_")}_final_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate and print basic metrics
        correct = sum(1 for r in results if r['prediction'].strip().lower() == r['actual'].strip().lower())
        accuracy = correct / len(results)
        print(f"\n{exp_name} Results:")
        print(f"Raw Accuracy (exact match): {accuracy:.4f}")
        print("\nNote: These results use exact string matching.")
        print("The original GPT-4 experiments used expert validation for semantic similarity.")
        print("Sample predictions vs actuals:")
        for i in range(min(3, len(results))):
            print(f"\nExample {i+1}:")
            print(f"Prediction: {results[i]['prediction']}")
            print(f"Actual: {results[i]['actual']}")
            print(f"Topics: {results[i]['training_topic']} -> {results[i]['test_topic']}")

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    main()
