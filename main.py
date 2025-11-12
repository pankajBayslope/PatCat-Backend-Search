# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=api_key)

app = FastAPI(title="AI Patent Search API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pat-cat-backend-search-wfrb.vercel.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@ app.on_event("startup")
async def load_data():
    global df
    try:
        df = pd.read_excel("patent_data.xlsx")
        df.fillna("", inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        df = pd.DataFrame()

# Request/Response Models
class SearchQuery(BaseModel):
    query: str

class PatentResult(BaseModel):
    Patent_Number: str
    Title: str
    Abstract: str
    Industry_Domain: str
    Technology_Area: str
    Sub_Technology_Area: str
    Keywords: str

# Extract Keywords
def extract_keywords_from_llm(query: str):
    prompt = f"""
    You are a professional patent search assistant.
    From the following user query, extract only the key technical or domain-specific keywords or phrases.
    Keep multi-word technical terms together (e.g., "3D imaging", "thermal imaging system").
    Ignore generic words like "show", "find", "patent", "related", "give me", etc.

    Query: "{query}"

    Return only the keywords or phrases separated by commas.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in patent keyword extraction."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=80,
        temperature=0.2,
    )
    keywords_text = response.choices[0].message.content.strip()
    keywords = [kw.strip().lower() for kw in re.split(r',\s*', keywords_text) if kw.strip()]
    return keywords

# Search Logic
def search_patents(keywords):
    if not keywords or df.empty:
        return []
    mask = pd.Series(False, index=df.index)
    for kw in keywords:
        pattern = re.escape(kw)
        kw_mask = (
            df["Industry Domain"].str.lower().str.contains(pattern, na=False) |
            df["Technology Area"].str.lower().str.contains(pattern, na=False) |
            df["Sub-Technology Area"].str.lower().str.contains(pattern, na=False) |
            df["Keywords"].str.lower().str.contains(pattern, na=False)
        )
        mask |= kw_mask
    return df[mask].to_dict(orient="records")

# API Routes
@app.get("/")
async def root():
    return {"message": "AI Patent Search API is running"}

@app.post("/search", response_model=dict)
async def search(request: SearchQuery):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    keywords = extract_keywords_from_llm(request.query)
    results = search_patents(keywords)

    return {
        "keywords": keywords,
        "results": results,
        "total": len(results)
    }