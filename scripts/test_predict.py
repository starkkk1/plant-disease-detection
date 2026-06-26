import requests
import os

def test_predict():
    url = "http://127.0.0.1:8000/predict"
    
    # Tạo một file ảnh tạm thời để test kết nối API (ảnh rỗng)
    # Lưu ý: Khi test thực tế, bạn nên thay đường dẫn bằng ảnh lá cây thật
    test_img_path = "test_dummy.jpg"
    from PIL import Image
    img = Image.new('RGB', (224, 224), color = 'green')
    img.save(test_img_path)
    
    try:
        print(f"Sending POST request to {url}")
        with open(test_img_path, "rb") as f:
            files = {"file": (test_img_path, f, "image/jpeg")}
            data = {"top_k": 3}
            response = requests.post(url, files=files, data=data)
            
        response.raise_for_status()
        
        result = response.json()
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"Trạng thái Model: {result.get('model_status')}")
        
        predictions = result.get("predictions", [])
        for idx, pred in enumerate(predictions):
            print(f"{idx+1}. Bệnh: {pred['class_name']} - Xác suất: {pred['probability']*100:.2f}%")
            
    except requests.exceptions.HTTPError as e:
        print(f"Lỗi API: {e.response.json().get('detail', str(e))}")
    except requests.exceptions.ConnectionError:
        print("Lỗi: Không thể kết nối tới server. Hãy chắc chắn bạn đã chạy 'python scripts/run_search_api.py'")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

if __name__ == "__main__":
    test_predict()
