from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

# Do not force Hugging Face offline in production; Render needs to download models on first use.
if os.environ.get("USE_HF_OFFLINE", "").lower() in {"1", "true", "yes"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import clip
from sklearn.cluster import KMeans

from transformers import pipeline
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from sentence_transformers import SentenceTransformer,util
print("CLIP Loaded:", clip.available_models())
import json
import numpy as np
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from llm_comparison import analyze_comparison_results, compare_embedding_results
from runtime_data import (
    data_path,
    get_clip_bundle,
    get_content_embeddings,
    get_content_embeddings2,
    get_content_embeddings3,
    get_full_reduced_corpus,
    get_german_hamlet_embeddings,
    get_hamlet_embeddings1,
    get_hamlet_embeddings2,
    get_hamlet_embeddings3,
    get_image_search_index,
    queryModel,
)

class SimilarityResult(BaseModel):
    topic_path: str
    content: str
    similarity_score: float
    rank: int

class SimilarityResponse(BaseModel):
    results: List[SimilarityResult]

class InputChunk(BaseModel):
    name: str
    value: str


class ClusterRequest(BaseModel):
    query: str
    k: int = 2
    chunks: List[InputChunk]
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY is not set. OpenAI endpoints will fail.")
openai_client = OpenAI(api_key=api_key) if api_key else None

ENGLISH_CHUNKS_JSON_PATH = str(data_path("english_chunks.json"))
GERMAN_CHUNKS_JSON_PATH = str(data_path("german_chunks.json"))
COMPARISON_RESULTS_JSON_PATH = str(data_path("comparison_results.json"))
EMBEDDING_COMPARISON_RESULTS_JSON_PATH = str(data_path("embedding_comparison_results.json"))
HAMLET_4X4_JSON_PATH = str(data_path("hamlet_4x4_comparison.json"))

app = FastAPI()

# app.mount("/images", StaticFiles(directory="images"), name="images")  
# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_Ai_data():
    try:
        with open('AI_wiki_embeddings.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")
    
def extract_content_and_embeddings(data, parent_path=""):
    results = []
    
    for item in data:
        current_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
        
        if 'content' in item and item['content'] and 'embeddings' in item:
            content = ''
            if 'content' in item:
                content = item['content']
            elif 'children' in item:
                content = concatenate_values(item['children'])
            else:
                content = item.value    

            results.append({
                'path': current_path,
                'content': content,
                'embeddings': item['embeddings']
            })
        else:
             
             content = item.get('value', '')  # Safely get 'value' if it exists
             results.append({
                'path': current_path,
                'content': item['value'],
                'embeddings': item.get('embeddings', None)
            })

        
        if 'children' in item:
            results.extend(extract_content_and_embeddings(item['children'], current_path))
    
    return results

PDF_FILE_PATH = str(data_path("History_of_artificial_intelligence.pdf"))

@app.get("/extract_paragraph/")
async def extract_paragraph_from_pdf(query: str = ""):
    """
    Extracts text from a local PDF file and returns the paragraph
    that best matches the provided query.
    """
    if not os.path.exists(PDF_FILE_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"The specified PDF file was not found at {PDF_FILE_PATH}"
        )

    try:
        # Step 1: Extract all paragraphs from the local PDF file
        paragraphs = []
        with open(PDF_FILE_PATH, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                # Split the text by double newline to get paragraphs
                paragraphs.extend(text.split('\n\n'))
        
        # Filter out any empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return {"message": "No text could be extracted from the PDF."}

        # If no query is provided, return the first paragraph or a default message
        if not query:
            return {"best_match": paragraphs[0]}

        # Step 2: Find the best matching paragraph
        # Use TF-IDF to vectorize the paragraphs and the query
        corpus = [query] + paragraphs
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

        # Calculate cosine similarity between the query vector and all paragraph vectors
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Find the index of the paragraph with the highest similarity
        best_match_index = cosine_sim.argmax()

        # Step 3: Return the best matching paragraph
        best_match_paragraph = paragraphs[best_match_index]
        return {"best_match": best_match_paragraph}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   
    
def extract_content_and_embeddings1(data, parent_path=""):
    results = []
    
    for item in data:
        current_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
        
        # Check if content and embeddings are present
        if 'content' in item and item['content'] and 'embeddings' in item:
            content = ''
            if 'content' in item:
                content = item['content']
            elif 'children' in item:
                content = concatenate_values(item['children'])
            else:
                content = item.get('value', '')  # Safely get 'value' if it exists
            
            results.append({
                'path': current_path,   
                'content': content,
                'embeddings': item['embeddings']
            })
        else:
            # Check for 'value' key and handle it safely using .get()
            content = item.get('value', '')  # Safely get 'value' if it exists
            
            results.append({
                'path': current_path,
                'content': content,
                'embeddings': item.get('embeddings', None)  # Default to None if 'embeddings' is missing
            })
        
        # Recursively process children if they exist
        if 'children' in item:
            results.extend(extract_content_and_embeddings(item['children'], current_path))
    
    return results

@app.post("/cluster-ai")
async def cluster_ai(request: ClusterRequest):
    try:
        # 1️⃣ Encode query
        query_embedding = queryModel.encode(request.query)
        query_embedding = torch.tensor(query_embedding).view(1, -1)

        chunk_embeddings = []
        similarities = []

        # 2️⃣ Encode all chunks
        for chunk in request.chunks:
            embedding = queryModel.encode(chunk.value)
            embedding_tensor = torch.tensor(embedding).view(1, -1)

            sim = util.cos_sim(query_embedding, embedding_tensor).item()

            chunk_embeddings.append(embedding)
            similarities.append(sim)

        chunk_embeddings = np.array(chunk_embeddings)

        # 3️⃣ KMeans clustering
        kmeans = KMeans(n_clusters=request.k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(chunk_embeddings)

        clusters = []
        best_cluster = None
        best_avg = -1

        for cluster_id in range(request.k):
            member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]

            member_chunks = [request.chunks[i] for i in member_indices]
            member_sims = [similarities[i] for i in member_indices]
            member_texts = [c.value for c in member_chunks]

            avg_similarity = float(np.mean(member_sims)) if member_sims else 0.0

            # TF-IDF themes
            if member_texts:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf = vectorizer.fit_transform(member_texts)
                scores = np.asarray(tfidf.mean(axis=0)).flatten()
                feature_names = np.array(vectorizer.get_feature_names_out())
                top_idx = scores.argsort()[-5:][::-1]
                themes = feature_names[top_idx].tolist()
            else:
                themes = []

            if avg_similarity > best_avg:
                best_avg = avg_similarity
                best_cluster = cluster_id

            clusters.append({
                "cluster_id": cluster_id,
                "members": [c.name for c in member_chunks],
                "average_similarity": avg_similarity,
                "semantic_themes": themes
            })

        return {
            "clusters": clusters,
            "best_cluster": best_cluster
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_hamlet_embeddings(data, parent_path=""):
    """Extract content and embeddings from Hamlet version files"""
    results = []
    
    for i, item in enumerate(data):
        # Handle cases where 'name' field might not exist
        if 'name' in item:
            current_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
        else:
            # Generate a name based on index if 'name' is missing
            item_name = f"Item_{i+1}"
            current_path = f"{parent_path}/{item_name}" if parent_path else item_name
        
        # Check if content and embeddings are present (note: singular 'embeddings')
        if 'content' in item and item['content'] and 'embeddings' in item:
            results.append({
                'path': current_path,
                'content': item['content'],
                'embeddings': item['embeddings']  # Convert 'embeddings' to 'embeddings' for consistency
            })
        
        # Recursively process sub_chapters if they exist
        if 'sub_chapters' in item:
            results.extend(extract_hamlet_embeddings(item['sub_chapters'], current_path))
        
        # Recursively process paragraphs if they exist
        if 'paragraphs' in item:
            results.extend(extract_hamlet_embeddings(item['paragraphs'], current_path))
        
        # Recursively process children if they exist
        if 'children' in item:
            results.extend(extract_hamlet_embeddings(item['children'], current_path))
    
    return results


def concatenate_values(data):
    return " ".join(item["value"] for item in data)

def transform_data(data):
    transformed = {"embeddings": []}
    
    for cluster in data["clusters"].values():
        for item in cluster:
            transformed["embeddings"].extend(item["embeddings"])
    
    return transformed

class QueryRequest(BaseModel):
    text: str
    chunks: int

# for item in content_embeddings3:
#     print(f"Path: {item['path']}")
#     print(f"embeddings shape: {torch.tensor(item['embeddings']).shape}")
#     print(f"First few values: {item['embeddings'][:5]}")  # Print sample
#     break  # Check only the first one
@app.post("/find-similar-AI", response_model=SimilarityResponse)
async def find_similar_AI(query: QueryRequest):
    try:
        # Generate embeddings for the query
        query_embedding = queryModel.encode(query.text)
        size = query.chunks

        query_embedding = torch.tensor(query_embedding).view(1, -1)

        # 🔥 Define this BEFORE loop
        query_lower = query.text.lower()

        similarities = []

        for item in get_content_embeddings3():
            if not item.get('embeddings'):
                continue

            try:
                stored_embedding = torch.tensor(item['embeddings']).view(1, -1)

                # Base similarity
                similarity = util.cos_sim(query_embedding, stored_embedding).item()

                # 🔥 HARD BOOST LOGIC
                if any(k in query_lower for k in [
                    "main people",
                    "who were",
                    "early development",
                    "beginning of artificial intelligence"
                ]):
                    if "dartmouth workshop" in item['content'].lower():
                        similarity += 0.03   # boost score

                similarities.append({
                    'topic_path': item['path'],
                    'content': item['content'],
                    'similarity_score': float(similarity) * 100
                })

            except Exception as e:
                print(f"Error processing item at path {item['path']}: {e}")
                continue

        # Sort results
        top_results = sorted(
            similarities,
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:size]

        # Add rank
        for i, result in enumerate(top_results):
            result['rank'] = size - i

        return SimilarityResponse(results=top_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def load_shakespeare_data():
    try:
        with open('version1_Shakes_final.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")

def load_shakespeare_data2():
    try:
        with open('version2_Shakes_final.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")

def load_hamlet_version1():
    try:
        with open('hamlet_version1.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Hamlet version 1 embeddings file not found")

def load_hamlet_version2():
    try:
        with open('hamlet_version2.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Hamlet version 2 embeddings file not found")

def load_hamlet_version3():
    try:
        with open('hamlet_version3.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Hamlet version 3 embeddings file not found")

def load_german_hamlet_data():
    try:
        with open('german_hamlet_embeddings.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("German Hamlet embeddings file not found")


# Hamlet / Shakespeare corpora load lazily via runtime_data.py

# for item in content_embeddings:
#     print(f"Path: {item['path']}")
#     print(f"embeddings shape: {torch.tensor(item['embeddings']).shape}")
#     print(f"First few values: {item['embeddings'][:5]}")  # Print sample
    # break  # Check only the first one
@app.post("/find-similar", response_model=SimilarityResponse)
async def find_similar(query: QueryRequest):
    try:
        # Generate embeddings for the query
        
        query_embedding = queryModel.encode(query.text)
        size = query.chunks
        query_embedding = torch.tensor(query_embedding).view(1, -1)  # Ensure (1, 384)
        # Calculate similarity scores
        similarities = []
        for item in get_content_embeddings():
           if not item.get('embeddings'):  # Skip if embeddings are None or empty
              continue

           try:
              stored_embedding = torch.tensor(item['embeddings']).view(1, -1)
              similarity = util.cos_sim(query_embedding, stored_embedding).item()

              similarities.append({
              'topic_path': item['path'],
              'content': item['content'],
              'similarity_score': float(similarity) * 100
               })
           except Exception as e:
               print(f"Error processing item at path {item['path']}: {e}")
               continue
       
        # Sort by similarity score and get top 10
        top_results = sorted(
            similarities, 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )[:size]
        
        # Add rank to top 10 results (10 = highest, 1 = lowest)
        for i, result in enumerate(top_results):
            result['rank'] = size - i
        
        return SimilarityResponse(results=top_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-hamlet1", response_model=SimilarityResponse)
async def find_similar_hamlet1(query: QueryRequest):
    try:
        # Generate embeddings for the query
        query_embedding = queryModel.encode(query.text)
        size = query.chunks
        query_embedding = torch.tensor(query_embedding).view(1, -1)  # Ensure (1, 384)
        
        # Calculate similarity scores
        similarities = []
        for item in get_hamlet_embeddings1():
           if not item.get('embeddings'):  # Skip if embeddings are None or empty
              continue

           try:
              stored_embedding = torch.tensor(item['embeddings']).view(1, -1)
              similarity = util.cos_sim(query_embedding, stored_embedding).item()

              similarities.append({
              'topic_path': item['path'],
              'content': item['content'],
              'similarity_score': float(similarity) * 100
               })
           except Exception as e:
               print(f"Error processing item at path {item['path']}: {e}")
               continue

        # Sort by similarity score and get top results
        top_results = sorted(
            similarities, 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )[:size]
        
        # Add rank to top results
        for i, result in enumerate(top_results):
            result['rank'] = size - i
        
        return SimilarityResponse(results=top_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-german-hamlet", response_model=SimilarityResponse)
async def find_similar_german_hamlet(query: QueryRequest):
    try:
        query_embedding = queryModel.encode(query.text)
        size = query.chunks
        query_embedding = torch.tensor(query_embedding).view(1, -1)

        similarities = []
        for item in get_german_hamlet_embeddings():
           if not item.get('embeddings'):
              continue

           try:
              stored_embedding = torch.tensor(item['embeddings']).view(1, -1)
              similarity = util.cos_sim(query_embedding, stored_embedding).item()

              similarities.append({
              'topic_path': item['path'],
              'content': item['content'],
              'similarity_score': float(similarity) * 100
               })
           except Exception as e:
               print(f"Error processing item at path {item['path']}: {e}")
               continue

        top_results = sorted(
            similarities,
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:size]

        for i, result in enumerate(top_results):
            result['rank'] = size - i

        return SimilarityResponse(results=top_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-hamlet2", response_model=SimilarityResponse)
async def find_similar_hamlet2(query: QueryRequest):
    try:
        # Generate embeddings for the query
        query_embedding = queryModel.encode(query.text)
        size = query.chunks
        query_embedding = torch.tensor(query_embedding).view(1, -1)  # Ensure (1, 384)
        
        # Calculate similarity scores
        similarities = []
        for item in get_hamlet_embeddings2():
           if not item.get('embeddings'):  # Skip if embeddings are None or empty
              continue

           try:
              stored_embedding = torch.tensor(item['embeddings']).view(1, -1)
              similarity = util.cos_sim(query_embedding, stored_embedding).item()

              similarities.append({
              'topic_path': item['path'],
              'content': item['content'],
              'similarity_score': float(similarity) * 100
               })
           except Exception as e:
               print(f"Error processing item at path {item['path']}: {e}")
               continue

        # Sort by similarity score and get top results
        top_results = sorted(
            similarities, 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )[:size]
        
        # Add rank to top results
        for i, result in enumerate(top_results):
            result['rank'] = size - i
        
        return SimilarityResponse(results=top_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-hamlet3", response_model=SimilarityResponse)
async def find_similar_hamlet3(query: QueryRequest):
    try:
        # Generate embeddings for the query
        query_embedding = queryModel.encode(query.text)
        size = query.chunks
        query_embedding = torch.tensor(query_embedding).view(1, -1)  # Ensure (1, 384)
        
        # Calculate similarity scores
        similarities = []
        for item in get_hamlet_embeddings3():
           if not item.get('embeddings'):  # Skip if embeddings are None or empty
              continue

           try:
              stored_embedding = torch.tensor(item['embeddings']).view(1, -1)
              similarity = util.cos_sim(query_embedding, stored_embedding).item()

              similarities.append({
              'topic_path': item['path'],
              'content': item['content'],
              'similarity_score': float(similarity) * 100
               })
           except Exception as e:
               print(f"Error processing item at path {item['path']}: {e}")
               continue

        # Sort by similarity score and get top results
        top_results = sorted(
            similarities, 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )[:size]
        
        # Add rank to top results
        for i, result in enumerate(top_results):
            result['rank'] = size - i
        
        return SimilarityResponse(results=top_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest2(BaseModel):
    query: str
    source: str
    german: bool = False
    language: str = ""

class KeywordRequest(BaseModel):
    query: str
    content: str
    top_n: int = 10

class KeywordResponse(BaseModel):
    keywords: List[str]

# @app.post("/extract_keywords", response_model=KeywordResponse)
# def extract_keywords(data: KeywordRequest):
#     keywords = extract_keywords_from_text(data.query, data.content, data.top_n)
#     return {"keywords": keywords}   
# Define request body
class SummaryRequest(BaseModel):
    query: str
    content: str


class EmbeddingComparisonRequest(BaseModel):
    english_query: str
    german_query: str
    english_paragraph: str
    german_paragraph: str
    model: str = "gpt-5"


class EmbeddingComparisonHighlight(BaseModel):
    phrase: str
    dimension: str


class EmbeddingComparisonLanguageDetail(BaseModel):
    matchedConcepts: List[str]
    reasoning: str
    chronologyScore: int
    semanticScore: int
    emotionScore: int
    styleScore: int
    metaphorScore: int
    toneScore: int
    voiceScore: int
    culturalScore: int
    highlights: List[EmbeddingComparisonHighlight]


class EmbeddingComparisonDifference(BaseModel):
    category: str
    english: str
    german: str
    impact: str


class EmbeddingComparisonResponse(BaseModel):
    summary: str
    english: EmbeddingComparisonLanguageDetail
    german: EmbeddingComparisonLanguageDetail
    differences: List[EmbeddingComparisonDifference]
    finalConclusion: str


class EmbeddingRetrievalComparisonRequest(BaseModel):
    english_query: str
    german_query: str
    model: str = "gpt-5"


class EmbeddingRetrievalComparisonResponse(BaseModel):
    english_query: str
    german_query: str
    english_chunk_id: str
    german_chunk_id: str
    english_paragraph: str
    german_paragraph: str
    analysis: EmbeddingComparisonResponse

summarizer = None

def get_summarizer():
    """Load BART lazily so uvicorn startup does not fail while importing app.py."""
    global summarizer
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=True,
        )
    return summarizer

@app.post("/summarize")
def summarize(request: SummaryRequest):
    query = request.query.strip()
    content = request.content.strip()

    if not query or not content:
        raise HTTPException(status_code=400, detail="Both 'query' and 'content' fields are required and non-empty.")

    # Simple way to bias content using query (prepend query to content)
    biased_input = f"{query}: {content}"

    try:
        summary = get_summarizer()(biased_input, max_length=130, min_length=30, do_sample=False)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar2", response_model=SimilarityResponse)
async def find_similar2(query: QueryRequest):
    try:
        # Generate embeddings for the query
        query_embedding = queryModel.encode(query.text)
        size = query.chunks
        query_embedding = torch.tensor(query_embedding).view(1, -1)  # Ensure (1, 384)
        if len(query_embedding.shape) == 3:
               query_embedding = query_embedding.squeeze(0)  # Remove extra dimension

        query_embedding = query_embedding.view(1, -1)
        
        # Calculate similarity scores
        similarities = []
        for item in get_content_embeddings2():
           if not item.get('embeddings'):  # Skip if embeddings are None or empty
              continue

           try:
              stored_embedding = torch.tensor(item['embeddings']).view(1, -1)
              similarity = util.cos_sim(query_embedding, stored_embedding).item()

              similarities.append({
              'topic_path': item['path'],
              'content': item['content'],
              'similarity_score': float(similarity) * 100
               })
           except Exception as e:
               print(f"Error processing item at path {item['path']}: {e}")
               continue
    
        # Sort by similarity score and get top 10
        top_results = sorted(
            similarities, 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )[:size]

        for i, result in enumerate(top_results):
            result['rank'] = size - i
        
        return SimilarityResponse(results=top_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_chunk_corpus(path: str) -> list:
    """Load flat chunk records from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _retrieve_top_chunk(query: str, chunks: list) -> dict:
    """Retrieve the top matching chunk using precomputed embeddings."""
    query_embedding = queryModel.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    query_vec = np.asarray(query_embedding, dtype=np.float64).ravel()

    best_chunk = None
    best_score = -1.0

    for chunk in chunks:
        embedding = chunk.get("embeddings")
        if not embedding:
            continue
        chunk_vec = np.asarray(embedding, dtype=np.float64).ravel()
        denom = np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
        score = float(np.dot(query_vec, chunk_vec) / denom) if denom > 1e-12 else 0.0
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None:
        raise HTTPException(status_code=404, detail="No retrievable chunks found")
    return best_chunk


@app.post("/compare-embedding-results", response_model=EmbeddingComparisonResponse)
async def compare_embedding_results_api(request: EmbeddingComparisonRequest):
    """
    Explain why the embedding model likely preferred different paragraphs
    for semantically equivalent English and German queries.
    """
    try:
        result = compare_embedding_results(
            english_query=request.english_query.strip(),
            german_query=request.german_query.strip(),
            english_paragraph=request.english_paragraph.strip(),
            german_paragraph=request.german_paragraph.strip(),
            client=openai_client,
            model=request.model,
        )
        return EmbeddingComparisonResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/compare-embedding-retrieval",
    response_model=EmbeddingRetrievalComparisonResponse,
)
async def compare_embedding_retrieval_api(request: EmbeddingRetrievalComparisonRequest):
    """
    Retrieve top English and German chunks, then explain why the embedding
    model preferred different paragraphs for equivalent queries.
    """
    try:
        english_chunks = _load_chunk_corpus(ENGLISH_CHUNKS_JSON_PATH)
        german_chunks = _load_chunk_corpus(GERMAN_CHUNKS_JSON_PATH)

        english_chunk = _retrieve_top_chunk(request.english_query.strip(), english_chunks)
        german_chunk = _retrieve_top_chunk(request.german_query.strip(), german_chunks)

        analysis = compare_embedding_results(
            english_query=request.english_query.strip(),
            german_query=request.german_query.strip(),
            english_paragraph=english_chunk.get("content", ""),
            german_paragraph=german_chunk.get("content", ""),
            client=openai_client,
            model=request.model,
        )

        return EmbeddingRetrievalComparisonResponse(
            english_query=request.english_query,
            german_query=request.german_query,
            english_chunk_id=english_chunk.get("chunk_id", ""),
            german_chunk_id=german_chunk.get("chunk_id", ""),
            english_paragraph=english_chunk.get("content", ""),
            german_paragraph=german_chunk.get("content", ""),
            analysis=EmbeddingComparisonResponse(**analysis),
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-embedding-batch")
async def compare_embedding_batch_api(model: str = "gpt-5"):
    """
    Run embedding comparison analysis for all rows in comparison_results.json
    and save Angular-ready JSON to embedding_comparison_results.json.
    """
    try:
        with open(COMPARISON_RESULTS_JSON_PATH, "r", encoding="utf-8") as file:
            comparison_rows = json.load(file)

        payload = analyze_comparison_results(
            comparison_rows=comparison_rows,
            english_chunks_path=Path(ENGLISH_CHUNKS_JSON_PATH),
            german_chunks_path=Path(GERMAN_CHUNKS_JSON_PATH),
            client=openai_client,
            model=model,
        )

        with open(EMBEDDING_COMPARISON_RESULTS_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return {
            "count": len(payload.get("passages", [])),
            "output_file": EMBEDDING_COMPARISON_RESULTS_JSON_PATH,
            "results": payload,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui/comparison")
async def get_comparison_ui(refresh: bool = False):
    """
    Return Angular-ready comparison JSON.

    Loads embedding_comparison_results.json when present. If missing, builds
    the payload from comparison_results.json and chunk files without calling
    the LLM so the UI can bind to real paragraph values immediately.
    """
    output_path = Path(EMBEDDING_COMPARISON_RESULTS_JSON_PATH)
    try:
        if output_path.exists() and not refresh:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "passages" in payload:
                return payload

        with open(COMPARISON_RESULTS_JSON_PATH, "r", encoding="utf-8") as file:
            comparison_rows = json.load(file)

        payload = analyze_comparison_results(
            comparison_rows=comparison_rows,
            english_chunks_path=Path(ENGLISH_CHUNKS_JSON_PATH),
            german_chunks_path=Path(GERMAN_CHUNKS_JSON_PATH),
            run_llm=False,
        )
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    except FileNotFoundError as e: 
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-ai/")
async def ask_ai(request: QueryRequest2):
    """
    Answer a query using only the most recent context in short.
    """
    try:
      
        query = request.query
       
        source_knowledge = request.source
        is_german = request.german or request.language.strip().lower() in {"german", "de", "deutsch"}
        
        # Build the prompt with only the current query's context
        # source_knowledge = "\n".join([x for x in results])
        if is_german:
            augmented_prompt = f"""Using the German context below, answer the query as a short summary in both English and German.
Keep each language to 2–3 sentences.

Return exactly this format:
English:
<English summary>

German:
<German summary>

        Context:
        {source_knowledge}

        Query: {query}"""
        else:
            augmented_prompt = f"""Using the context below, answer the query briefly (2–3 sentences max).

        Context:
        {source_knowledge}

        Query: {query}"""


        chat_messages = [
            SystemMessage(content="You are a helpful assistant."),  # Role definition
            HumanMessage(content=augmented_prompt),                 # Current query with context
        ]
        
        # Generate the response
        # chat = ChatOpenAI(
        #     openai_api_key=api_key,
        #     model='gpt-3.5-turbo'
        # )
        chat = ChatOpenAI(
            openai_api_key=api_key,
            model='gpt-4o'
        )
       
        response = chat(chat_messages)
        return {"query": query, "response": response.content}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
async def health():
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if methods and path:
            for method in sorted(methods - {"HEAD", "OPTIONS"}):
                routes.append(f"{method} {path}")
    return {
        "status": "ok",
        "service": "vidoreader-api",
        "deployed_module": "app.py",
        "data_root": str(data_path(".")),
        "routes": sorted(routes),
        "openai_configured": openai_client is not None,
    }


@app.get("/hamlet/4x4-comparison")
async def get_hamlet_4x4_comparison():
    path = Path(HAMLET_4X4_JSON_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail="hamlet_4x4_comparison.json not found")
    return json.loads(path.read_text(encoding="utf-8"))


class ImageSet(BaseModel):
    images: List[str]  # file paths
    prompt: str

# CLIP + SentenceTransformer load lazily via runtime_data.py
FULL_REDUCED_LEAF_ONLY = True
FULL_REDUCED_SCALE_SCORES = True


def _fr_cosine_numpy(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    q = np.asarray(query_vec, dtype=np.float64).ravel()
    d = np.asarray(doc_vec, dtype=np.float64).ravel()
    denom = np.linalg.norm(q) * np.linalg.norm(d)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(q, d) / denom)


def _fr_node_label(item: dict) -> str:
    return (
        item.get("name")
        or item.get("chapter_title")
        or item.get("subchapter_title")
        or item.get("chunk_name")
        or "unnamed"
    )


def _fr_node_text(item: dict) -> str:
    v = item.get("value") or item.get("content")
    if v:
        return v if isinstance(v, str) else str(v)
    return ""


def _fr_child_list(item: dict) -> list:
    return item.get("children") or item.get("subchapters") or []


def _fr_concatenate_values(children: list) -> str:
    if not children:
        return ""
    parts = []
    for child in children:
        if isinstance(child, dict):
            t = _fr_node_text(child)
            if t:
                parts.append(t)
            nested = _fr_child_list(child)
            if nested:
                sub = _fr_concatenate_values(nested)
                if sub:
                    parts.append(sub)
    return "\n\n".join(parts)


def _fr_extract_flat(data, parent_path: str = "") -> list:
    """Flatten Full_reduced-style tree (chapters / children / value / embeddings)."""
    results = []
    if isinstance(data, dict):
        if "chapters" in data:
            return _fr_extract_flat(data["chapters"], parent_path)
        data = [data]
    if not isinstance(data, list):
        return results
    for item in data:
        if not isinstance(item, dict):
            continue
        label = _fr_node_label(item)
        current_path = f"{parent_path}/{label}" if parent_path else str(label)
        content = _fr_node_text(item)
        kids = _fr_child_list(item)
        if not content and kids:
            content = _fr_concatenate_values(kids)
        emb = item.get("embeddings")
        if emb is None:
            emb = item.get("embedding")
        results.append(
            {
                "path": current_path,
                "content": content,
                "embeddings": emb,
                "is_leaf": not bool(kids),
            }
        )
        if kids:
            results.extend(_fr_extract_flat(kids, current_path))
    return results


@app.post("/find-similar-full-reduced", response_model=SimilarityResponse)
async def find_similar_full_reduced(query: QueryRequest):
    """
    Semantic search over the Full_reduced management textbook
    (precomputed embeddings in Full_reduced_structured_with_embeddings.json).
    """
    full_reduced_corpus = get_full_reduced_corpus(leaf_only=FULL_REDUCED_LEAF_ONLY)
    if not full_reduced_corpus:
        raise HTTPException(
            status_code=503,
            detail="Corpus not loaded. Place Full_reduced_structured_with_embeddings.json in DATA_ROOT.",
        )
    try:
        size = max(1, int(query.chunks))
        query_vec = queryModel.encode(
            query.text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        similarities = []
        for item in full_reduced_corpus:
            try:
                raw = _fr_cosine_numpy(query_vec, item["embeddings"])
                similarities.append(
                    {
                        "topic_path": item["path"],
                        "content": item["content"],
                        "raw_cosine": raw,
                    }
                )
            except Exception as e:
                print(f"Full_reduced skip {item.get('path')}: {e}")
                continue
        if not similarities:
            raise HTTPException(
                status_code=500,
                detail="No similarity scores computed for any corpus item.",
            )
        if FULL_REDUCED_SCALE_SCORES:
            raw_vals = [s["raw_cosine"] for s in similarities]
            lo, hi = min(raw_vals), max(raw_vals)
            span = hi - lo
            for s in similarities:
                if span > 1e-12:
                    s["similarity_score"] = (s["raw_cosine"] - lo) / span * 100.0
                else:
                    s["similarity_score"] = 100.0
        else:
            for s in similarities:
                s["similarity_score"] = s["raw_cosine"] * 100.0
        top_results = sorted(
            similarities,
            key=lambda x: x["similarity_score"],
            reverse=True,
        )[:size]
        out = []
        for i, row in enumerate(top_results):
            out.append(
                SimilarityResult(
                    topic_path=row["topic_path"],
                    content=row["content"],
                    similarity_score=round(float(row["similarity_score"]), 4),
                    rank=size - i,
                )
            )
        return SimilarityResponse(results=out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Load stored embeddings
# with open("merged_output.json", "r") as f:
#     image_data = json.load(f)

# image_embeddings = np.array([item['embeddings'] for item in image_data])
# image_paths = [item['path'] for item in image_data]
# umap_x = [item['umap_x'] for item in image_data]
# umap_y = [item['umap_y'] for item in image_data]
# bird_names = [item['folder'] for item in image_data]

# print(bird_names,38)

@app.get("/search")
def search(query: str = Query(..., description="Text query for CLIP search")):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    index = get_image_search_index()
    image_embeddings = index["embeddings"]
    if image_embeddings.size == 0:
        raise HTTPException(
            status_code=503,
            detail="clip_image_embeddings.json not found in DATA_ROOT.",
        )

    model, _preprocess, device = get_clip_bundle()
    with torch.no_grad():
        text_token = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text_token).cpu().numpy()

    similarities = cosine_similarity(text_embedding, image_embeddings)[0]
    top_k = similarities.argsort()[::-1][:20]
    image_paths = index["paths"]
    bird_names = index["folders"]
    umap_x = index["umap_x"]
    umap_y = index["umap_y"]

    results = [
        {
            "image": image_paths[i].split("images")[-1],
            "folder": bird_names[i],
            "umap_x": umap_x[i],
            "umap_y": umap_y[i],
            "score": float(similarities[i]),
        }
        for i in top_k
    ]

    return JSONResponse({"images": results})
