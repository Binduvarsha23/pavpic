import torch
import numpy as np
import clip
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from PIL import Image
import uvicorn

# FastAPI instance
app = FastAPI()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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

# Function to extract image embeddings from a file
def get_image_embedding(image_file):
    image = Image.open(BytesIO(image_file))
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input).cpu().numpy().flatten()
    return image_embedding

# Endpoint to get embeddings for both image and text
@app.post("/get_embeddings")
async def get_embeddings(image: UploadFile = File(...), query_text: str = ""):
    try:
        # Get image embedding
        image_data = await image.read()
        image_embedding = get_image_embedding(image_data)
        
        # Get text embedding if text is provided
        text_embedding = None
        if query_text:
            text_embedding = get_text_embedding(query_text)
        
        return {"image_embedding": image_embedding.tolist(), "text_embedding": text_embedding.tolist() if text_embedding is not None else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to find top N similar images based on text query
def search_images(query_text, threshold=0.25, top_n=6):
    query_embedding = get_text_embedding(query_text).reshape(1, -1)  # Reshape to 2D array

    matches = []

    # Assuming image_db contains pre-calculated embeddings
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

port = int(os.environ.get("PORT", 80))

# Run the app with Uvicorn for local testing (when not using Render)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
