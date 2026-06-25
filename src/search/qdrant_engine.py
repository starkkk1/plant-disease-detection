import os
import uuid
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: qdrant_client could not be imported: {e}")
    QDRANT_AVAILABLE = False

class MultimodalSearchEngine:
    def __init__(self, db_path: str = "data/qdrant_db"):
        """
        Khởi tạo kết nối Qdrant Client (local file-based).
        """
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant_client is not available due to an import error.")
        os.makedirs(db_path, exist_ok=True)
        self.client = QdrantClient(path=db_path)
        print(f"Connected to Qdrant at {db_path}")

    def create_collection(self, collection_name: str, vector_size: int = 512):
        """
        Tạo mới một Collection. Nếu đã tồn tại, sẽ bỏ qua hoặc ghi đè (ở script build).
        """
        try:
            # Kiểm tra xem collection đã tồn tại chưa
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name in collections:
                print(f"Collection '{collection_name}' already exists. Recreating...")
                self.client.delete_collection(collection_name=collection_name)
                
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created collection '{collection_name}' with vector size {vector_size}")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def insert_embeddings(self, collection_name: str, embeddings: list, payloads: list):
        """
        Thêm các vector và metadata vào Collection.
        """
        points = []
        for i, (emb, payload) in enumerate(zip(embeddings, payloads)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Inserted {len(points)} points into '{collection_name}'")

    def search(self, collection_name: str, query_vector: list, limit: int = 5):
        """
        Tìm kiếm K điểm gần nhất với query_vector.
        """
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=limit
        )
        return results.points
