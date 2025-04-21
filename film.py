import os
import json
import sqlite3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import httpx
import asyncio

load_dotenv()
DB_NAME = "movies.db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

encoder = SentenceTransformer(EMBEDDING_MODEL)

def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        director TEXT,
        year INTEGER,
        genre TEXT,
        lead_actor TEXT,
        duration INTEGER,
        embedding BLOB
    )
    """)
    
    conn.commit()
    conn.close()

def load_movies():
    with open("movies.json", "r", encoding="utf-8") as f:
        movies = json.load(f)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM movies")
    
    for movie in movies:
        text = f"{movie['title']} {movie['director']} {movie['genre']} {movie['lead_actor']}"
        vector = encoder.encode(text)
        
        cursor.execute("""
        INSERT INTO movies (title, director, year, genre, lead_actor, duration, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            movie["title"],
            movie["director"],
            movie["year"],
            movie["genre"],
            movie["lead_actor"],
            movie["duration"],
            vector.tobytes()
        ))
    
    conn.commit()
    conn.close()
    
    return movies

def search_movies_semantic(query: str, top_k: int = 3, genre_filter: Optional[str] = None) -> List[Dict]:
    query_vector = encoder.encode(query)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title, director, year, genre, lead_actor, duration, embedding FROM movies")
    movies = cursor.fetchall()
    
    results = []
    for movie in movies:
        id_, title, director, year, genre, lead_actor, duration, embedding_bytes = movie
        
        if genre_filter and genre != genre_filter:
            continue
            
        stored_vector = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        similarity = np.dot(query_vector, stored_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(stored_vector))
        
        results.append({
            "title": title,
            "director": director,
            "year": year,
            "genre": genre,
            "lead_actor": lead_actor,
            "duration": duration,
            "score": float(similarity)
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    conn.close()
    return results[:top_k]

async def semantic_search_with_llm(query: str, movies_data: list, top_k: int = 3) -> List[Dict]:
    context = "\n".join(
        f"{idx}. {m['title']} ({m['year']}, {m['director']}): {m['genre']} starring {m['lead_actor']} | {m['duration']} min"
        for idx, m in enumerate(movies_data, 1)
    )
    
    prompt = f"""Analyze the list of movies and find {top_k} most relevant for the query:
    
    Query: {query}
    
    Available movies:
    {context}
    
    Return JSON with indexes of {top_k} most relevant movies in order of relevance.
    Format: {{"results": [idx1, idx2, idx3]}}"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            indices = json.loads(result["choices"][0]["message"]["content"])["results"]
            return [movies_data[i-1] for i in indices]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

async def main():
    initialize_database()
    movies_data = load_movies()
    
    while True:
        query = input("\nSearch or 'exit': ")
        if query.lower() == 'exit':
            break
            
        genre_filter = input("Filter by genre (leave empty for no filter): ").strip()
        if not genre_filter:
            genre_filter = None
            
        print("\nPerforming semantic search...")
        results = search_movies_semantic(query, genre_filter=genre_filter)
        
        print("\nPerforming LLM-enhanced search...")
        llm_results = await semantic_search_with_llm(query, movies_data)
        
        combined = {}
        for movie in results + llm_results:
            if movie["title"] not in combined:
                combined[movie["title"]] = movie
        
        final_results = list(combined.values())[:5]
        
        if not final_results:
            print("No movies found")
        else:
            print("\nResults:")
            for idx, movie in enumerate(final_results, 1):
                print(f"{idx}. {movie['title']} ({movie['year']})")
                print(f"   Director: {movie['director']}")
                print(f"   Genre: {movie['genre']}")
                print(f"   Starring: {movie['lead_actor']}")
                print(f"   Duration: {movie['duration']} minutes")
                if 'score' in movie:
                    print(f"   Similarity score: {movie['score']:.3f}")
                print()

if __name__ == "__main__":
    asyncio.run(main())