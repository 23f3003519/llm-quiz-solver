import requests
import json

# Test data - UPDATE THESE VALUES!
test_data = {
    "email": "your_email@example.com",
    "secret": "mysecret2024",  # Must match your .env SECRET_STRING
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

def test_api():
    try:
        print("Testing API endpoint...")
        
        # Test local server
        response = requests.post(
            "http://localhost:8000/solve",
            json=test_data,
            timeout=180
        )
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()