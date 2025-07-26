import streamlit as st
import torch
import clip
from PIL import Image
import os
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CLIP Image Search", layout="wide")

# Load CLIP model
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess

model, preprocess = load_model()

# Store image features
image_db = []

def extract_features(images):
    features = []
    for img_path, img in images:
        preprocessed = preprocess(img).unsqueeze(0).to("cpu")
        with torch.no_grad():
            feature = model.encode_image(preprocessed)
        feature /= feature.norm(dim=-1, keepdim=True)
        features.append((img_path, feature.cpu().numpy()))
    return features

def search_images(query, features):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize([query]).to("cpu"))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        query_feat = text_encoded.cpu().numpy()

    results = []
    for img_path, feat in features:
        score = cosine_similarity(query_feat, feat)[0][0]
        results.append((img_path, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

# UI
st.title("🔍 CLIP Image Search (ViT-B/32)")
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} images uploaded ✅")
    images = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        images.append((uploaded_file.name, img))
    
    st.info("Extracting features... 🔍")
    image_db = extract_features(images)

    query = st.text_input("Enter search prompt (e.g., 'a cat wearing sunglasses')")

    if query and image_db:
        st.subheader("🔎 Search Results")
        results = search_images(query, image_db)[:12]  # top 12 results

        cols = st.columns(4)
        for idx, (img_path, score) in enumerate(results):
            with cols[idx % 4]:
                st.image(dict(images)[img_path], caption=f"{img_path} ({score:.2f})", use_column_width=True)
