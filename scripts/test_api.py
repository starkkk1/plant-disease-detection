import requests
import json

def test_text_search():
    url = "http://127.0.0.1:8000/search/text"
    payload = {
        "query": "Bệnh đốm lá cà chua",
        "limit": 3
    }
    
    try:
        print(f"Sending POST request to {url} with query: '{payload['query']}'")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        results = response.json().get("results", [])
        print(f"\nFound {len(results)} results:")
        for idx, res in enumerate(results):
            score = res.get("score")
            payload_data = res.get("payload", {})
            class_name = payload_data.get("class_name", "Unknown")
            print(f"{idx+1}. Score: {score:.4f} | Class: {class_name}")
            
    except requests.exceptions.ConnectionError:
        print("Lỗi: Không thể kết nối tới server. Hãy chắc chắn rằng bạn đã chạy 'python scripts/run_search_api.py'")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    test_text_search()
