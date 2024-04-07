import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Retrieve the API token and URL from environment variables
API_TOKEN = os.getenv("API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"

# Set up the authorization headers
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def test_gemma_api_response():
    """Test that the Hugging Face API for the gemma-7b-it model returns a valid response."""
    # Define a sample payload
    sample_payload = {
        "inputs": "Can you please let us know more details about your project?"
    }

    # Make a POST request to the API
    response = requests.post(API_URL, headers=headers, json=sample_payload)

    # Check that the response status code is 200 (OK)
    assert response.status_code == 200, "API did not return a successful response."

    # Check that the response contains a list of dictionaries with a 'generated_text' key
    response_data = response.json()
    assert isinstance(response_data, list), "API response is not a list."
    assert len(response_data) > 0, "API response list is empty."
    assert (
        "generated_text" in response_data[0]
    ), "First item in API response list does not contain 'generated_text' key."


# If you're running this file directly, execute the test
if __name__ == "__main__":
    test_gemma_api_response()
