import requests
import json

# Test data -  FOR LIVE DEPLOYMENT
test_data = {
    "email": "test@example.com",
    "secret": "mysecret2024",  
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

def test_api():
    try:
        print("Testing LIVE API endpoint...")
        
        # Test LIVE Vercel server (not localhost)
        response = requests.post(
            "https://llm-quiz-solver-pttyi00ac-23f3003519s-projects.vercel.app/solve",  
            json=test_data,
            timeout=180
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()