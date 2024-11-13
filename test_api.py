import os
import json
import sys
from openai import OpenAI

def test_openai_api():
    try:
        # Debug: Print all available keys
        print("\nAvailable environment variables for OpenAI:")
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            print(f"Found OPENAI_API_KEY: {api_key[:4]}...{api_key[-4:]}")
        else:
            print("OPENAI_API_KEY not found")
            return False

        print(f"\nUsing API key: {api_key[:4]}...{api_key[-4:]}")

        client = OpenAI(api_key=api_key)

        # Test GPT-4o
        print("\nTesting GPT-4o API connection...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.2,
            max_tokens=100
        )
        print("✓ GPT-4o API Connection Successful")
        print(f"Response: {response.choices[0].message.content}")

        return True
    except Exception as e:
        print(f"❌ API Test Failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Full error details: {sys.exc_info()}")
        return False

def test_data_access():
    try:
        # Test data.json access
        with open('data/data.json', 'r') as f:
            data = json.load(f)
        print("\nData Access Test Results:")
        print("✓ data.json loaded successfully")
        print(f"✓ Found {len(data)} entries")
        return True
    except Exception as e:
        print(f"❌ Data Access Failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running Repository Tests...")

    # Debug: Print environment variables
    print("\nEnvironment Variables Check:")
    api_key = os.environ.get('OPENAI_API_KEY')
    print(f"OPENAI_API_KEY set: {'Yes' if api_key else 'No'}")

    api_success = test_openai_api()
    data_success = test_data_access()

    if api_success and data_success:
        print("\nAll tests passed successfully! ✓")
    else:
        print("\nSome tests failed. Please check the errors above. ❌")
