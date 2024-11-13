import os
import json
import random
import time
import base64
from pathlib import Path
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_results(results, filename):
    with open(f'outputs/{filename}', 'w') as f:
        json.dump(results, f, indent=2)

def get_image_path(image_ref: str) -> str:
    if not image_ref:
        return None

    base_path = Path('data images')
    for ext in ['.jpg', '.png']:
        for subdir in ['questions', 'learner answer', 'correct answer']:
            potential_path = base_path / subdir / f"{image_ref}{ext}"
            if potential_path.exists():
                return str(potential_path)
    return None

def encode_image(image_path: str) -> str:
    if not image_path:
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def prepare_message_content(example, include_images=True):
    content = []

    content.append({"type": "text", "text": f"Question: {example['Question']}"})
    if include_images and example.get('Question image'):
        image_path = get_image_path(example['Question image'])
        if image_path:
            base64_image = encode_image(image_path)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

    content.append({"type": "text", "text": f"Incorrect Answer: {example['Incorrect Answer']}"})
    if include_images and example.get('Learner Answer image'):
        image_path = get_image_path(example['Learner Answer image'])
        if image_path:
            base64_image = encode_image(image_path)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

    return content

def get_gpt4o_response(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            messages = [{"role": "system", "content": "You are an expert in identifying mathematical misconceptions."}]

            if isinstance(prompt, str):
                # Text-only prompt
                messages.append({"role": "user", "content": prompt})
            else:
                # Multi-modal prompt with potential images
                messages.append({"role": "user", "content": prepare_message_content(prompt)})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
                frequency_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def run_experiment_1(dataset, batch_size=10):
    """Cross-Topic Testing with image support"""
    results = []
    topics = {}

    # Group examples by topic
    for example in dataset:
        topic = example['Topic']
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(example)

    # Select one random example from each topic for training
    training_examples = []
    for topic_examples in topics.values():
        training_examples.append(random.choice(topic_examples))

    # Create training prompt
    training_prompt = "Here are some examples of mathematical misconceptions:\n\n"
    for ex in training_examples:
        training_prompt += f"Topic: {ex['Topic']}\nQuestion: {ex['Question']}\nIncorrect Answer: {ex['Incorrect Answer']}\nMisconception: {ex['Misconception']}\n\n"

    # Test on random examples from entire dataset
    test_examples = random.sample([ex for ex in dataset if ex not in training_examples], 100)

    for i, test_example in enumerate(test_examples):
        print(f"\nProcessing example {i+1}/100 for Cross-Topic Testing")
        print(f"Topic: {test_example['Topic']}")

        # Pass the example as a dictionary to support images
        prediction = get_gpt4o_response({
            'Topic': test_example['Topic'],
            'Question': test_example['Question'],
            'Incorrect Answer': test_example['Incorrect Answer'],
            'Question image': test_example.get('Question image'),
            'Learner Answer image': test_example.get('Learner Answer image')
        })

        if prediction:
            results.append({
                'topic': test_example['Topic'],
                'question': test_example['Question'],
                'incorrect_answer': test_example['Incorrect Answer'],
                'actual_misconception': test_example['Misconception'],
                'predicted_misconception': prediction,
                'has_images': bool(test_example.get('Question image') or
                                 test_example.get('Learner Answer image'))
            })

        if (i + 1) % batch_size == 0:
            print(f"Batch {(i + 1) // batch_size} completed")
            save_results(results, f'gpt4o_experiment_1_(cross-topic)_results_{i+1}.json')
            print(f"Results saved for {i+1} examples")

    save_results(results, 'gpt4o_experiment_1_(cross-topic)_final_results.json')
    return results

def run_experiment_2(dataset, batch_size=10):
    """Topic-Constrained Testing with image support"""
    results = []
    topics = {}

    # Group examples by topic
    for example in dataset:
        topic = example['Topic']
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(example)

    # For each topic, select examples for training and testing
    all_test_examples = []
    for topic_examples in topics.values():
        if len(topic_examples) >= 2:  # Need at least 2 examples per topic
            training_example = random.choice(topic_examples)
            test_examples = [ex for ex in topic_examples if ex != training_example]
            test_example = random.choice(test_examples)
            all_test_examples.append({
                'training': training_example,
                'test': test_example
            })

    # Randomly sample to get 100 test cases
    test_cases = random.sample(all_test_examples, min(100, len(all_test_examples)))

    for i, case in enumerate(test_cases):
        training_example = case['training']
        test_example = case['test']

        print(f"\nProcessing example {i+1}/{len(test_cases)} for Topic-Constrained Testing")
        print(f"Topic: {test_example['Topic']}")

        # Pass the example as a dictionary to support images
        prediction = get_gpt4o_response({
            'Topic': test_example['Topic'],
            'Question': test_example['Question'],
            'Incorrect Answer': test_example['Incorrect Answer'],
            'Question image': test_example.get('Question image'),
            'Learner Answer image': test_example.get('Learner Answer image')
        })

        if prediction:
            results.append({
                'topic': test_example['Topic'],
                'question': test_example['Question'],
                'incorrect_answer': test_example['Incorrect Answer'],
                'actual_misconception': test_example['Misconception'],
                'predicted_misconception': prediction,
                'has_images': bool(test_example.get('Question image') or
                                 test_example.get('Learner Answer image'))
            })

        if (i + 1) % batch_size == 0:
            print(f"Batch {(i + 1) // batch_size} completed")
            save_results(results, f'gpt4o_experiment_2_(topic-constrained)_results_{i+1}.json')
            print(f"Results saved for {i+1} examples")

    save_results(results, 'gpt4o_experiment_2_(topic-constrained)_final_results.json')
    return results

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    # Load dataset from data/data.json
    print("Loading dataset from data/data.json...")
    dataset = load_dataset('data/data.json')

    # Run experiments
    print("Running Experiment 1 (Cross-Topic Testing)...")
    experiment_1_results = run_experiment_1(dataset)
    print(f"Completed Experiment 1 with {len(experiment_1_results)} results")

    print("\nRunning Experiment 2 (Topic-Constrained Testing)...")
    experiment_2_results = run_experiment_2(dataset)
    print(f"Completed Experiment 2 with {len(experiment_2_results)} results")
