import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pytrends.request import TrendReq

load_dotenv()

app = FastAPI(
    title="Valuelytics GBP Audit API",
    version="1.0.0",
    description="Google Business Profile audit using Google Trends and Google Maps competitor intelligence."
)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# -----------------------------
# Models
# -----------------------------
class BusinessInput(BaseModel):
    business_name: str = Field(
        ...,
        example="Petra Mediterranean Restaurant",
        description="Official business name as shown on Google Maps"
    )
    address: str = Field(
        ...,
        example="Tampa, FL",
        description="City or full address used for location-based search"
    )
    category: str = Field(
        ...,
        example="Mediterranean restaurant",
        description="Primary Google Business Profile category"
    )


class Competitor(BaseModel):
    name: Optional[str] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    category: Optional[str] = None
    address: Optional[str] = None


class ScoreBreakdown(BaseModel):
    base: int = 100
    review_gap_penalty: int = 0
    rating_gap_penalty: int = 0
    amenities_penalty: int = 10
    description_penalty: int = 10
    photo_penalty: int = 5
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


# -----------------------------
# Utilities
# -----------------------------
def _require_serpapi_key() -> None:
    if not SERPAPI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="SERPAPI_API_KEY is not set. Add it to your .env file and restart the server."
        )


def _normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _serpapi_get(params: Dict[str, Any]) -> Dict[str, Any]:
    _require_serpapi_key()

    params = dict(params)
    params["api_key"] = SERPAPI_API_KEY

    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream API error (SerpAPI): {str(e)}")


# -----------------------------
# Google Maps: Business lookup and competitors
# -----------------------------
def find_business_on_maps(business_name: str, location: str) -> Dict[str, Any]:
    """
    Finds the most likely business listing on Google Maps via SerpAPI.
    Returns a dict with rating, reviews, title, address and raw fields if available.
    """
    data = _serpapi_get({
        "engine": "google_maps",
        "type": "search",
        "q": f"{business_name} {location}"
    })

    results = data.get("local_results", []) or []

    if not results:
        return {
            "found": False,
            "title": None,
            "rating": None,
            "reviews": None,
            "address": None,
            "raw": {}
        }

    target_norm = _normalize_name(business_name)

    best = results[0]
    best_score = -1

    for item in results[:10]:
        title = item.get("title") or ""
        title_norm = _normalize_name(title)

        score = 0
        if title_norm == target_norm:
            score += 10
        elif target_norm and title_norm and (target_norm in title_norm or title_norm in target_norm):
            score += 6

        addr = (item.get("address") or "").lower()
        loc = (location or "").lower()
        if loc and addr and (loc.split(",")[0].strip() in addr):
            score += 2

        if score > best_score:
            best = item
            best_score = score

    return {
        "found": True,
        "title": best.get("title"),
        "rating": _safe_float(best.get("rating")),
        "reviews": _safe_int(best.get("reviews")),
        "address": best.get("address"),
        "raw": best
    }


def get_google_maps_competitors(category: str, location: str, limit: int = 5) -> List[Dict[str, Any]]:
    data = _serpapi_get({
        "engine": "google_maps",
        "type": "search",
        "q": f"{category} near {location}"
    })

    competitors: List[Dict[str, Any]] = []
    for place in (data.get("local_results", []) or [])[: max(limit, 1)]:
        competitors.append({
            "name": place.get("title"),
            "rating": _safe_float(place.get("rating")),
            "reviews": _safe_int(place.get("reviews")),
            "category": place.get("type"),
            "address": place.get("address")
        })
    return competitors


def exclude_self_from_competitors(competitors: List[Dict[str, Any]], business_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (filtered_competitors, removed_self_matches)
    """
    bn = _normalize_name(business_name)
    filtered: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for c in competitors:
        cn = _normalize_name(c.get("name") or "")
        if bn and cn and (bn == cn or bn in cn or cn in bn):
            removed.append(c)
        else:
            filtered.append(c)

    return filtered, removed


def competitor_benchmarks(competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not competitors:
        return {
            "count": 0,
            "avg_reviews": 0,
            "avg_rating": 0
        }

    ratings = [c["rating"] for c in competitors if isinstance(c.get("rating"), (int, float))]
    reviews = [c["reviews"] for c in competitors if isinstance(c.get("reviews"), int)]

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0
    avg_reviews = int(sum(reviews) / len(reviews)) if reviews else 0

    return {
        "count": len(competitors),
        "avg_reviews": avg_reviews,
        "avg_rating": avg_rating
    }


# -----------------------------
# Scoring
# -----------------------------
def calculate_optimization_score(
    competitors_for_benchmark: List[Dict[str, Any]],
    business_rating: Optional[float],
    business_reviews: Optional[int]
) -> ScoreBreakdown:
    base = 100

    # Baseline penalties until you pull actual GBP fields (amenities, description, photos)
    amenities_penalty = 10
    description_penalty = 10
    photo_penalty = 5

    review_gap_penalty = 0
    rating_gap_penalty = 0

    bm = competitor_benchmarks(competitors_for_benchmark)
    avg_reviews = bm["avg_reviews"]
    avg_rating = bm["avg_rating"]

    # Review gap penalty up to 30
    if business_reviews is not None and avg_reviews and avg_reviews > 0:
        gap_ratio = max(0.0, (avg_reviews - business_reviews) / avg_reviews)
        review_gap_penalty = min(30, int(round(gap_ratio * 30)))

    # Rating gap penalty up to 30
    if business_rating is not None and avg_rating and avg_rating > 0:
        rating_gap = max(0.0, avg_rating - business_rating)
        rating_gap_penalty = min(30, int(round(rating_gap * 20)))

    final_score = base - amenities_penalty - description_penalty - photo_penalty - review_gap_penalty - rating_gap_penalty
    final_score = max(0, min(100, final_score))

    return ScoreBreakdown(
        base=base,
        review_gap_penalty=review_gap_penalty,
        rating_gap_penalty=rating_gap_penalty,
        amenities_penalty=amenities_penalty,
        description_penalty=description_penalty,
        photo_penalty=photo_penalty,
        final_score=final_score
    )


# -----------------------------
# Trends
# -----------------------------
def get_trends_snapshot(category: str) -> Dict[str, Any]:
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(
            [category, f"{category} near me"],
            timeframe="today 12-m",
            geo="US"
        )
        return pytrends.interest_over_time().tail(1).to_dict()
    except Exception as e:
        # Trends should not break the entire audit
        return {"error": f"Google Trends unavailable: {str(e)}"}


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def health():
    return {"status": "Valuelytics GBP Audit API running", "docs": "/docs"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(data: BusinessInput):
    if not data.business_name.strip() or not data.address.strip() or not data.category.strip():
        raise HTTPException(status_code=400, detail="business_name, address, and category are required.")

    # 1) Find the business metrics on Maps (rating, reviews)
    biz = find_business_on_maps(data.business_name, data.address)

    business_rating = biz.get("rating")
    business_reviews = biz.get("reviews")

    # 2) Pull competitors
    raw_competitors = get_google_maps_competitors(data.category, data.address, limit=7)

    # 3) Exclude the business itself from benchmark calculations
    competitors_filtered, removed_self = exclude_self_from_competitors(raw_competitors, data.business_name)

    # 4) Benchmarks and score
    bm = competitor_benchmarks(competitors_filtered)
    breakdown = calculate_optimization_score(
        competitors_for_benchmark=competitors_filtered,
        business_rating=business_rating,
        business_reviews=business_reviews
    )

    # 5) Trends snapshot
    trends = get_trends_snapshot(data.category)

    # 6) Recommendations, now tied to real benchmarks
    recommendations: List[str] = []

    if bm["count"] > 0:
        if business_reviews is not None and bm["avg_reviews"] > 0:
            if business_reviews < bm["avg_reviews"]:
                recommendations.append(
                    f"Your listing has {business_reviews} reviews. Top competitors average {bm['avg_reviews']}. Increasing review volume is recommended."
                )
            else:
                recommendations.append(
                    f"Your review volume is competitive. You have {business_reviews} reviews vs competitor average {bm['avg_reviews']}."
                )
        else:
            recommendations.append(
                f"Top competitors average {bm['avg_reviews']} reviews. Increasing review volume is recommended."
            )

        if business_rating is not None and bm["avg_rating"] > 0:
            if business_rating < bm["avg_rating"]:
                recommendations.append(
                    f"Your rating is {business_rating}. Top competitors average {bm['avg_rating']}. Aim to exceed this benchmark."
                )
            else:
                recommendations.append(
                    f"Your rating is competitive at {business_rating} vs competitor average {bm['avg_rating']}."
                )
        else:
            recommendations.append(
                f"Top competitors average a {bm['avg_rating']} rating. Aim to exceed this benchmark."
            )

    # Client-ready baseline actions (until you add true GBP field parsing)
    recommendations.extend([
        "Ensure all relevant amenities are selected in the profile",
        "Optimize business description with category and location keywords",
        "Add fresh photos weekly to improve engagement and ranking"
    ])

    # 7) Build response
    competitors_out = [Competitor(**c) for c in raw_competitors]

    business_metrics = {
        "found_on_maps": bool(biz.get("found")),
        "matched_title": biz.get("title"),
        "matched_address": biz.get("address"),
        "rating": business_rating,
        "reviews": business_reviews,
        "excluded_from_benchmarks_count": len(removed_self)
    }

    return AnalyzeResponse(
        business=data,
        business_metrics=business_metrics,
        optimization_score=breakdown.final_score,
        score_breakdown=breakdown,
        trends=trends,
        competitors=competitors_out,
        competitor_benchmarks=bm,
        recommendations=recommendations
    )


# Optional convenience endpoint for simple clients that prefer query strings
@app.get("/analyze", response_model=AnalyzeResponse)
def analyze_get(
    business_name: str = Query(..., description="Business name as shown on Google Maps"),
    address: str = Query(..., description="City or full address"),
    category: str = Query(..., description="Primary GBP category")
):
    return analyze(BusinessInput(business_name=business_name, address=address, category=category))