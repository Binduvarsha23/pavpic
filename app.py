import torch
import numpy as np
import json
import os
import clip
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# FastAPI instance
app = FastAPI()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load embeddings from JSON file
json_filename = "embeddings.json"
with open(json_filename, 'r') as json_file:
    image_db = json.load(json_file)

# Convert embeddings back to numpy arrays
for img_path, embedding_list in image_db.items():
    image_db[img_path] = np.array(embedding_list)

# Pydantic model for the search query
class SearchQuery(BaseModel):
    query_text: str
    threshold: float = 0.25
    top_n: int = 6

# Function to extract text embeddings
def get_text_embedding(text):
    text_tokenized = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text_tokenized).cpu().numpy().flatten()
    return embedding

# Function to find top N similar images based on text query
def search_images(query_text, threshold=0.25, top_n=6):
    query_embedding = get_text_embedding(query_text).reshape(1, -1)  # Reshape to 2D array

    matches = []

    for path, stored_embedding in image_db.items():
        stored_embedding = stored_embedding.reshape(1, -1)  # Reshape to 2D array
        similarity = cosine_similarity(query_embedding, stored_embedding)[0][0]
        if similarity > threshold:
            matches.append((path, similarity))  # Store matches with similarity > threshold

    # Sort matches by similarity score in descending order and select top_n
    matches.sort(key=lambda x: x[1], reverse=True)
    top_matches = matches[:top_n]

    return top_matches

# FastAPI endpoint to search for images
@app.post("/search")
async def search(query: SearchQuery):
    try:
        matches = search_images(query.query_text, threshold=query.threshold, top_n=query.top_n)
        return {"matches": [{"image_path": match[0], "similarity": match[1]} for match in matches]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
port = int(os.environ.get("PORT", 8000))
# Run the app with Uvicorn for local testing (when not using Render)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
