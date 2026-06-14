import os
import urllib.request
import json
import urllib.parse
from pathlib import Path

def download_wikimedia_images(query, num_images, out_dir):
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageimages&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={num_images}&piprop=original"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_info in pages.items():
                if 'original' in page_info:
                    img_url = page_info['original']['source']
                    img_name = page_info['title'].split(':')[-1]
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        out_path = os.path.join(out_dir, img_name.replace(' ', '_'))
                        print(f"Downloading {img_name}...")
                        try:
                            req_img = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req_img) as response_img:
                                with open(out_path, 'wb') as f:
                                    f.write(response_img.read())
                        except Exception as e:
                            print(f"Failed to download {img_url}: {e}")
    except Exception as e:
        print(f"Failed to search for {query}: {e}")

out_dir = r"d:\Code\python\plant-disease-detection\data\backgrounds"
os.makedirs(out_dir, exist_ok=True)

download_wikimedia_images("grass texture", 10, out_dir)
download_wikimedia_images("soil texture", 10, out_dir)
