import os
import requests

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pytrends.request import TrendReq

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Read API key from environment
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# -----------------------------
# Helper: Google Maps Competitors
# -----------------------------
def get_google_maps_competitors(category: str, location: str):
    if not SERPAPI_API_KEY:
        print("SERPAPI_API_KEY not set")
        return []

    params = {
        "engine": "google_maps",
        "q": f"{category} near {location}",
        "type": "search",
        "api_key": SERPAPI_API_KEY
    }

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("SerpAPI error:", e)
        return []

    competitors = []

    for place in data.get("local_results", [])[:5]:
        competitors.append({
            "name": place.get("title"),
            "rating": place.get("rating"),
            "reviews": place.get("reviews"),
            "category": place.get("type"),
            "address": place.get("address")
        })

    return competitors


# -----------------------------
# Request Model
# -----------------------------
class BusinessInput(BaseModel):
    business_name: str
    address: str
    category: str


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "Valuelytics GBP Audit API running"}


# -----------------------------
# Main Analysis Endpoint
# -----------------------------
@app.post("/analyze")
def analyze(data: BusinessInput):
    # Google Trends
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload(
        [data.category, f"{data.category} near me"],
        timeframe="today 12-m",
        geo="US"
    )

    trends = pytrends.interest_over_time().tail(1).to_dict()

    # Competitors
    competitors = get_google_maps_competitors(
        data.category,
        data.address
    )

    # Recommendations
    recommendations = []

    if competitors:
        avg_reviews = sum(c["reviews"] or 0 for c in competitors) / len(competitors)
        avg_rating = sum(c["rating"] or 0 for c in competitors) / len(competitors)

        recommendations.append(
            f"Top competitors average {int(avg_reviews)} reviews. Increasing review volume is recommended."
        )

        recommendations.append(
            f"Top competitors average a {round(avg_rating, 1)} rating. Aim to exceed this benchmark."
        )

    recommendations.extend([
        "Ensure all relevant amenities are selected",
        "Optimize business description with category keywords",
        "Add fresh photos weekly to improve engagement"
    ])

    return {
        "business": data.dict(),
        "trends": trends,
        "competitors": competitors,
        "recommendations": recommendations
    }
