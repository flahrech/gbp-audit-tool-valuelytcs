import os
import re
import sqlite3
import requests
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pytrends.request import TrendReq
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

load_dotenv()

# --- Initialization ---
app = FastAPI(title="Valuelytics GBP Audit API", version="1.2.0")
templates = Jinja2Templates(directory="templates")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
LEADS_DB_PATH = os.getenv("LEADS_DB_PATH") or "leads.db"

# --- Models ---
class BusinessInput(BaseModel):
    business_name: str
    address: str
    category: str

class Competitor(BaseModel):
    name: Optional[str]
    rating: Optional[float]
    reviews: Optional[int]
    category: Optional[str]
    address: Optional[str]

class ScoreBreakdown(BaseModel):
    base: int = 100
    review_gap_penalty: int = 0
    rating_gap_penalty: int = 0
    amenities_penalty: int = 0
    description_penalty: int = 0
    photo_penalty: int = 0
    final_score: int = 0

class AnalyzeResponse(BaseModel):
    business: BusinessInput
    business_metrics: Dict[str, Any]
    optimization_score: int
    score_breakdown: ScoreBreakdown
    trends: Dict[str, Any]
    competitors: List[Competitor]
    competitor_benchmarks: Dict[str, Any]
    recommendations: List[str]

# --- Database Helpers ---
def init_leads_db():
    conn = sqlite3.connect(LEADS_DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS leads (id INTEGER PRIMARY KEY, created_at TEXT, email TEXT, business_name TEXT, address TEXT, category TEXT)")
    conn.close()

def save_lead(email: str, biz: BusinessInput):
    conn = sqlite3.connect(LEADS_DB_PATH)
    conn.execute("INSERT INTO leads (created_at, email, business_name, address, category) VALUES (?, ?, ?, ?, ?)",
                 (datetime.utcnow().isoformat(), email, biz.business_name, biz.address, biz.category))
    conn.commit()
    conn.close()

init_leads_db()

# --- Utility Helpers ---
def _safe_int(x): return int(x) if x else 0
def _safe_float(x): return float(x) if x else 0.0
def _normalize(s): return re.sub(r"[^a-z0-9 ]+", "", s.lower().strip())

def _serpapi_get(params):
    params["api_key"] = SERPAPI_API_KEY
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# --- Core Logic: Scanning & Analysis ---

def find_business_on_maps(name: str, location: str) -> Dict[str, Any]:
    """Scans for the target business and extracts specific optimization data."""
    data = _serpapi_get({"engine": "google_maps", "type": "search", "q": f"{name} {location}"})
    results = data.get("local_results", [])
    if not results:
        return {"found": False, "rating": 0, "reviews": 0, "has_description": False, "photo_count": 0, "amenities_count": 0}

    best = results[0] # Usually the most relevant match
    return {
        "found": True,
        "title": best.get("title"),
        "rating": _safe_float(best.get("rating")),
        "reviews": _safe_int(best.get("reviews")),
        "address": best.get("address"),
        "has_description": bool(best.get("description") or best.get("snippet")),
        "photo_count": len(best.get("photos", [])),
        "amenities_count": len(best.get("service_options", []) or best.get("attributes", []) or [])
    }

def get_competitors(category: str, location: str) -> List[Dict[str, Any]]:
    """Pulls local rivals based on category."""
    data = _serpapi_get({"engine": "google_maps", "type": "search", "q": f"{category} in {location}"})
    return [{
        "name": p.get("title"),
        "rating": _safe_float(p.get("rating")),
        "reviews": _safe_int(p.get("reviews")),
        "category": p.get("type"),
        "address": p.get("address")
    } for p in data.get("local_results", [])[:10]]

def calculate_score(bm: Dict[str, Any], biz: Dict[str, Any]) -> ScoreBreakdown:
    """Calculates penalties based on real scanned metrics."""
    rev_gap = min(30, int(((bm["avg_reviews"] - biz["reviews"]) / bm["avg_reviews"] * 30))) if bm["avg_reviews"] > biz["reviews"] else 0
    rat_gap = min(20, int((bm["avg_rating"] - biz["rating"]) * 20)) if bm["avg_rating"] > biz["rating"] else 0
    
    desc_pen = 10 if not biz["has_description"] else 0
    amen_pen = 10 if biz["amenities_count"] < 3 else 0
    photo_pen = 10 if biz["photo_count"] < 5 else (5 if biz["photo_count"] < 10 else 0)

    final = 100 - rev_gap - rat_gap - desc_pen - amen_pen - photo_pen
    return ScoreBreakdown(
        review_gap_penalty=rev_gap, rating_gap_penalty=rat_gap,
        description_penalty=desc_pen, amenities_penalty=amen_pen,
        photo_penalty=photo_pen, final_score=max(0, final)
    )

def run_audit(business: BusinessInput) -> AnalyzeResponse:
    biz_data = find_business_on_maps(business.business_name, business.address)
    comps = get_competitors(business.category, business.address)
    
    # Exclude self from benchmarks
    filtered_comps = [c for c in comps if _normalize(business.business_name) not in _normalize(c["name"])]
    
    # Benchmarking
    avg_rating = round(sum(c["rating"] for c in filtered_comps) / len(filtered_comps), 1) if filtered_comps else 0
    avg_reviews = int(sum(c["reviews"] for c in filtered_comps) / len(filtered_comps)) if filtered_comps else 0
    bm = {"avg_rating": avg_rating, "avg_reviews": avg_reviews, "count": len(filtered_comps)}

    breakdown = calculate_score(bm, biz_data)

    recs = [
        f"Increase reviews to match market average of {avg_reviews}.",
        "Improve rating to exceed local benchmark of " + str(avg_rating) + "."
    ]
    if not biz_data["has_description"]: recs.append("Add a keyword-rich business description.")
    if biz_data["photo_count"] < 10: recs.append("Upload at least 10 high-resolution photos.")

    return AnalyzeResponse(
        business=business,
        business_metrics=biz_data,
        optimization_score=breakdown.final_score,
        score_breakdown=breakdown,
        trends={}, # Optional: Google Trends integration
        competitors=[Competitor(**c) for c in filtered_comps],
        competitor_benchmarks=bm,
        recommendations=recs
    )

# --- Routes ---

@app.get("/app", response_class=HTMLResponse)
async def serve_app(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

@app.post("/app/analyze", response_class=HTMLResponse)
async def analyze_app(request: Request, business_name: str = Form(...), address: str = Form(...), category: str = Form(...), email: str = Form("")):
    biz = BusinessInput(business_name=business_name, address=address, category=category)
    payload = run_audit(biz)
    if email: save_lead(email, biz)
    return templates.TemplateResponse("results.html", {"request": request, **payload.dict()})

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_api(data: BusinessInput):
    return run_audit(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)