import os
import sys
import uvicorn

# Đảm bảo import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Starting Plant Disease Search API on http://0.0.0.0:8000")
    # Sử dụng reload=True cho quá trình development
    uvicorn.run("src.search.api:app", host="0.0.0.0", port=8000, reload=True)
