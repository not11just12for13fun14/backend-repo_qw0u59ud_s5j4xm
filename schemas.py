"""
Database Schemas for Food Recognition App

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class FoodItem(BaseModel):
    name: str = Field(..., description="Detected food name (normalized)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence 0-1")
    weight_grams: Optional[float] = Field(None, ge=0, description="Estimated weight in grams if available")
    calories: Optional[float] = Field(None, ge=0, description="Calories for this item (kcal)")

class Analysis(BaseModel):
    image_url: Optional[str] = Field(None, description="Stored URL or data reference of the analyzed image")
    items: List[FoodItem] = Field(default_factory=list, description="List of detected food items with calories")
    total_calories: float = Field(0, ge=0, description="Total calories across all items")
    notes: Optional[str] = Field(None, description="Any extra info about the prediction")
