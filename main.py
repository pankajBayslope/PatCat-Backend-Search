# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import traceback

# ==================== ENVIRONMENT ====================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment variables!")

client = OpenAI(api_key=api_key)

app = FastAPI(title="AI Patent Search API", version="1.0")

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pat-cat-backend-search-wfrb.vercel.app",
        "http://localhost:3000",  # Local dev
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL DATA ====================
df = pd.DataFrame()  # Global DataFrame

@app.on_event("startup")
async def load_data():
    global df
    try:
        # ABSOLUTE PATH — Critical for Vercel
        current_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(current_dir, "patent_data.xlsx")

        print(f"[STARTUP] Looking for Excel file at: {excel_path}")

        if not os.path.exists(excel_path):
            error_msg = f"FILE NOT FOUND: {excel_path}"
            print(error_msg)
            df = pd.DataFrame()
            return

        print(f"[STARTUP] Loading Excel file...")
        df = pd.read_excel(excel_path, engine="openpyxl")
        df.fillna("", inplace=True)
        print(f"[SUCCESS] Loaded {len(df)} patents from Excel")

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[DATA LOAD FAILED] {e}\n{error_detail}")
        df = pd.DataFrame()

# ==================== MODELS ====================
class SearchQuery(BaseModel):
    query: str

# ==================== KEYWORD EXTRACTION ====================
def extract_keywords_from_llm(query: str):
    prompt = f"""
    You are a professional patent search assistant.
    From the following user query, extract only the key technical or domain-specific keywords or phrases.
    Keep multi-word technical terms together (e.g., "3D imaging", "thermal imaging system").
    Ignore generic words like "show", "find", "patent", "related", "give me", etc.

    Query: "{query}"

    Return only the keywords or phrases separated by commas.
    """

    try:
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
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return []

# ==================== SEARCH LOGIC ====================
def search_patents(keywords):
    if not keywords or df.empty:
        return []
    
    mask = pd.Series([False] * len(df), dtype=bool)
    for kw in keywords:
        pattern = re.escape(kw)
        kw_mask = (
            df["Industry Domain"].astype(str).str.lower().str.contains(pattern, na=False) |
            df["Technology Area"].astype(str).str.lower().str.contains(pattern, na=False) |
            df["Sub-Technology Area"].astype(str).str.lower().str.contains(pattern, na=False) |
            df["Keywords"].astype(str).str.lower().str.contains(pattern, na=False)
        )
        mask |= kw_mask
    return df[mask].to_dict(orient="records")

# ==================== ROUTES ====================
@app.get("/")
async def root():
    return {"message": "AI Patent Search API is running", "patents_loaded": len(df)}

@app.post("/search")
async def search(request: SearchQuery):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"[SEARCH] Query: {request.query}")
    keywords = extract_keywords_from_llm(request.query)
    results = search_patents(keywords)

    print(f"[RESULT] Keywords: {keywords} | Found: {len(results)} patents")

    return {
        "keywords": keywords,
        "results": results,
        "total": len(results)
    }

# ==================== GLOBAL ERROR HANDLER ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = traceback.format_exc()
    print(f"[SERVER ERROR] {exc}\n{error_detail}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
    )