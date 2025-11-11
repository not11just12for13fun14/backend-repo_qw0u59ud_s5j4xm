import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np

from database import db, create_document
from schemas import FoodItem, Analysis

# Simple label → calories per 100g reference (demo). In real apps, expand this DB.
CALORIES_PER_100G = {
    "apple": 52,
    "banana": 89,
    "orange": 47,
    "bread": 265,
    "rice": 130,
    "pasta": 131,
    "chicken": 239,
    "egg": 155,
    "milk": 42,
    "salad": 20,
    "tomato": 18,
    "potato": 77,
    "cheese": 402,
    "yogurt": 59,
    "pizza": 266,
    "burger": 295,
    "fries": 312,
    "soda": 41,
    "coffee": 1,
    "tea": 1,
}

# Very lightweight demo classifier: uses color/shape heuristics to guess likely foods.
# This avoids needing heavy ML dependencies while demonstrating backend-first flow.
# Replace with a real model later if desired.
def simple_food_classifier(img: Image.Image) -> List[FoodItem]:
    img_resized = img.convert("RGB").resize((64, 64))
    arr = np.array(img_resized) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))  # [r, g, b]
    r, g, b = mean_rgb

    candidates = []

    # Heuristic rules
    if r > g + 0.1 and r > b + 0.1:
        candidates.append(("tomato", 0.6))
        candidates.append(("apple", 0.4))
    if g > r + 0.1 and g > b + 0.05:
        candidates.append(("salad", 0.55))
        candidates.append(("apple", 0.25))
    if b > r + 0.05 and b > g + 0.05:
        candidates.append(("blueberry", 0.5))  # not in calorie map, will default

    # Fallback generic guesses based on brightness
    brightness = arr.mean()
    if brightness > 0.7:
        candidates.append(("rice", 0.35))
        candidates.append(("bread", 0.2))
    elif brightness < 0.3:
        candidates.append(("burger", 0.3))
        candidates.append(("fries", 0.25))

    # Deduplicate by highest confidence
    best = {}
    for name, conf in candidates:
        if name not in best or conf > best[name]:
            best[name] = conf

    # Convert to FoodItem list
    items: List[FoodItem] = []
    for name, conf in best.items():
        # Default estimated serving 100g
        cal_100g = CALORIES_PER_100G.get(name.lower())
        calories = None
        if cal_100g is not None:
            calories = float(cal_100g)  # for 100g default
        items.append(FoodItem(name=name.lower(), confidence=float(conf), weight_grams=100.0, calories=calories))

    # If nothing detected, return an empty list
    return items

class AnalysisResponse(BaseModel):
    items: List[FoodItem]
    total_calories: float
    message: str
    analysis_id: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Food Calorie Detection API is running"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read file bytes
        contents = await file.read()
        # Load image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run simple heuristic classifier
        detected_items = simple_food_classifier(image)

        # If no calories known, set 0
        total_calories = 0.0
        for it in detected_items:
            if it.calories is None:
                it.calories = 0.0
            total_calories += float(it.calories or 0)

        # Save analysis to DB
        analysis = Analysis(items=detected_items, total_calories=total_calories)
        try:
            analysis_id = create_document("analysis", analysis)
        except Exception as e:
            # Database may not be configured; still return result
            analysis_id = "db-disabled"

        return AnalysisResponse(
            items=detected_items,
            total_calories=round(total_calories, 2),
            message="Analysis complete",
            analysis_id=analysis_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
