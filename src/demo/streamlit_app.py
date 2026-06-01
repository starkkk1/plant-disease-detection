import os
import sys
import tempfile
import streamlit as st
from PIL import Image

# Đảm bảo import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.multimodal_encoder import MultimodalEncoder
from src.search.qdrant_engine import MultimodalSearchEngine

st.set_page_config(page_title="Plant Disease Search", layout="wide")

@st.cache_resource
def load_models():
    encoder = MultimodalEncoder()
    engine = MultimodalSearchEngine()
    return encoder, engine

encoder, engine = load_models()
COLLECTION_NAME = "tomato_disease_multimodal"

st.title("🌿 Multimodal Plant Disease Search")
st.markdown("Bạn có thể tìm kiếm bằng **Ảnh**, **Văn bản (Tiếng Việt)** hoặc **Kết hợp cả hai**!")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Đầu vào (Input)")
    
    upload_file = st.file_uploader("Upload ảnh lá cây (Tuỳ chọn)", type=["jpg", "png", "jpeg"])
    text_query = st.text_input("Nhập triệu chứng bệnh bằng Tiếng Việt (Tuỳ chọn)", placeholder="Ví dụ: lá bị đốm đen")
    
    alpha = st.slider("Mức độ ưu tiên Ảnh / Văn bản (Chỉ dùng khi nhập cả hai)", 
                      min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                      help="1.0 = Chỉ dùng ảnh, 0.0 = Chỉ dùng văn bản")
    
    search_button = st.button("Tìm kiếm", type="primary")
    
    if upload_file is not None:
        st.image(upload_file, caption="Ảnh bạn tải lên", width=300)

with col2:
    st.subheader("2. Kết quả (Top 5)")
    
    if search_button:
        if not upload_file and not text_query.strip():
            st.warning("Vui lòng cung cấp ít nhất một ảnh hoặc một câu mô tả.")
        else:
            with st.spinner("Đang tìm kiếm..."):
                query_vector = None
                
                # Case 1: Cả Ảnh + Text
                if upload_file and text_query.strip():
                    # Save tạm file ảnh
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(upload_file.getvalue())
                        tmp_path = tmp.name
                    
                    query_vector = encoder.encode_multimodal(tmp_path, text_query.strip(), alpha=alpha)
                    os.unlink(tmp_path)
                
                # Case 2: Chỉ Ảnh
                elif upload_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(upload_file.getvalue())
                        tmp_path = tmp.name
                        
                    query_vector = encoder.encode_image(tmp_path)
                    os.unlink(tmp_path)
                    
                # Case 3: Chỉ Text
                elif text_query.strip():
                    query_vector = encoder.encode_text(text_query.strip())
                
                if query_vector is not None:
                    # Query Qdrant
                    results = engine.search(COLLECTION_NAME, query_vector, limit=5)
                    
                    if not results:
                        st.info("Không tìm thấy kết quả nào. Có thể database chưa được build.")
                    else:
                        for i, res in enumerate(results):
                            st.markdown(f"**Top {i+1}** - Độ tương đồng: `{res.score:.4f}`")
                            st.markdown(f"🏷️ **Nhãn**: `{res.payload.get('class_name', 'Unknown')}`")
                            
                            img_path = res.payload.get('image_path')
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, width=250)
                            else:
                                st.error(f"Không tìm thấy file ảnh gốc: {img_path}")
                            st.divider()
