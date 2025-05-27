from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
model.encode("warmup", convert_to_tensor=True)  # warmup once on load
