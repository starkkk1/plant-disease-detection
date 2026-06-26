import os
import sys
import uvicorn

# Đảm bảo import được src và set working directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
os.chdir(project_root)

if __name__ == "__main__":
    print("Starting Plant Disease Search API on http://0.0.0.0:8000")
    # Sử dụng reload=True cho quá trình development
    uvicorn.run("src.search.api:app", host="0.0.0.0", port=8000, reload=True)
