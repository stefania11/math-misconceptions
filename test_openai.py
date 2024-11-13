import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OAI_key'))

# List available models
print("Available Models:")
models = client.models.list()
for model in models.data:
    print(model.id)

# Test API with a simple request
try:
    print("\nTesting API with simple request...")
    response = client.chat.completions.create(
        model="gpt-4",  # Try with standard GPT-4 first
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.2,
        max_tokens=10
    )
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
