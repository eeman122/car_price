from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Connection
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import joblib
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder

import os
os.environ["HF_HOME"] = "/tmp"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price Prediction API",
              description="API for car price prediction and analytics",
              version="1.0.0")

# CORS Configuration - Updated for Hugging Face deployment
# origins = [
#     "http://localhost:8501",  # For local testing
#     "http://127.0.0.1:8000",  # For local testing
#     "https://*.hf.space",  # Allows all Hugging Face Spaces
#     f"https://{os.getenv('HF_USERNAME', 'annus-lums')}-car-price-app-frontend.hf.space"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://car-price-1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite Database Connection
def get_db() -> Connection:
    """Get a SQLite database connection with optimized settings"""
    conn = sqlite3.connect('car_data.db')
    # Optimize for read performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

# Load ML models
try:
    linear_model = mlflow.sklearn.load_model("models/linear_model")
    poly_model = mlflow.sklearn.load_model("models/poly_model")
    logger.info("✅ ML models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load ML models: {e}")
    raise

# Load scalers
try:
    scalers = joblib.load("models/scalers.pkl")
    logger.info("✅ Scalers loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load scalers: {e}")
    raise

# Request models
class CarDetails(BaseModel):
    make: str
    model: str
    year: int
    engine: float
    fuel: str
    transmission: str
    mileage: int
    assembly: Optional[str] = None

class SearchParams(BaseModel):
    color: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

# Prediction endpoint
@app.post("/predict", summary="Predict car price")
def predict(details: CarDetails):
    try:
        logger.info("Model expects these features: %s", linear_model.feature_names_in_)
        input_data = details.dict()
        df = pd.DataFrame([input_data])
        logger.debug("Input data: %s", df.head())
        
        processed_data = preprocess_data(df, train_fit=False, scalers=scalers)
        logger.debug("Processed data features: %s", processed_data.columns.tolist())

        linear_pred = linear_model.predict(processed_data)[0]
        return {"prediction": round(linear_pred, 0)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Analytics endpoints
@app.get("/analytics/imported", summary="Get imported vs local car stats")
def get_imported_analytics():
    try:
        conn = get_db()
        query = """
            SELECT 
                CASE WHEN assembly = 'Imported' THEN 'Imported' ELSE 'Local' END AS assembly_type,
                COUNT(*) AS count,
                AVG(price) AS avg_price
            FROM car_data
            GROUP BY assembly_type
        """
        data = pd.read_sql(query, conn)
        conn.close()
        
        if data.empty:
            return [{"assembly_type": "Local", "count": 0, "avg_price": 0},
                   {"assembly_type": "Imported", "count": 0, "avg_price": 0}]
        return data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Imported analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/transmission", summary="Get transmission type stats")
def get_transmission_analytics():
    try:
        conn = get_db()
        query = """
            SELECT 
                transmission,
                COUNT(*) AS count,
                AVG(price) AS avg_price
            FROM car_data
            WHERE transmission IS NOT NULL
            GROUP BY transmission
        """
        data = pd.read_sql(query, conn)
        conn.close()
        
        if data.empty:
            return [{"transmission": "Automatic", "count": 0, "avg_price": 0},
                   {"transmission": "Manual", "count": 0, "avg_price": 0}]
        return data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Transmission analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.get("/search", summary="Search car listings")
def search_cars(
    color: Optional[str] = None,
    make: Optional[str] = None,
    model: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
):
    try:
        conn = get_db()
        query = "SELECT * FROM car_data WHERE 1=1"
        params = {}
        
        # Build the query
        if color:
            query += " AND color = :color"
            params["color"] = color
        if make:
            query += " AND make = :make"
            params["make"] = make
        if model:
            query += " AND model = :model"
            params["model"] = model
        if min_price is not None:
            query += " AND price >= :min_price"
            params["min_price"] = min_price
        if max_price is not None:
            query += " AND price <= :max_price"
            params["max_price"] = max_price
        
        # Execute query and handle results
        data = pd.read_sql(query, conn, params=params)
        conn.close()
        
        # Convert all numpy types to native Python types
        data = data.astype(object).where(pd.notnull(data), None)
        
        # Convert DataFrame to list of dicts with proper type handling
        results = []
        for _, row in data.iterrows():
            clean_row = {}
            for col, value in row.items():
                # Handle numpy types and NaN values
                if pd.isna(value):
                    clean_row[col] = None
                elif hasattr(value, 'item'):  # For numpy types
                    clean_row[col] = value.item()
                else:
                    clean_row[col] = value
            results.append(clean_row)
        
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")

def preprocess_data(df, train_fit=True, scalers=None):
    """
    Correct preprocessing that handles prediction without price column
    """
    # Use precomputed mappings during prediction
    if not train_fit:
        mileage_year_map = scalers['mileage_year_map']
        median_year = scalers['median_year']
        top_20_make_models = scalers['top_20_make_models']
        most_common_fuel = scalers.get("most_common_fuel", "Petrol")
        most_common_transmission = scalers.get("most_common_transmission", "Manual")

    # Handle missing values
    df['year'] = df['year'].fillna(median_year)
    df['engine'] = df['engine'].fillna(df['engine'].median())
    df['mileage'] = df['mileage'].fillna(df['mileage'].median())
    
    # Calculate age
    df['age'] = (2025 - df['year']) + 1
    
    # Handle assembly encoding
    df['assembly_encoded'] = df['assembly'].apply(
        lambda x: 1 if str(x).lower() == 'local' else 0
    )
    
    # Create make_model feature (consistent formatting)
    df['make'] = df['make'].str.lower().str.strip()
    df['model'] = df['model'].str.lower().str.strip()
    df['make_model'] = df['make'] + '_' + df['model']
    
    # List of all expected MM_ columns from your model
    expected_mm_columns = [
        'MM_Daihatsu_Cuore', 'MM_Daihatsu_Mira', 'MM_Honda_City',
        'MM_Honda_Civic', 'MM_Honda_Vezel', 'MM_KIA_Sportage',
        'MM_Suzuki_Alto', 'MM_Suzuki_Bolan', 'MM_Suzuki_Cultus',
        'MM_Suzuki_Mehran', 'MM_Suzuki_Swift', 'MM_Suzuki_Wagon',
        'MM_Toyota_Corolla', 'MM_Toyota_Fortuner', 'MM_Toyota_Hilux',
        'MM_Toyota_Land', 'MM_Toyota_Passo', 'MM_Toyota_Prado',
        'MM_Toyota_Vitz', 'MM_Toyota_Yaris', 'MM_Unknown'
    ]
    
    # Initialize all MM columns to 0
    for col in expected_mm_columns:
        df[col] = 0
    
    # Set the appropriate column to 1
    for idx, row in df.iterrows():
        mm_col = 'MM_' + row['make_model'].title().replace(' ', '_')
        if mm_col in expected_mm_columns:
            df.at[idx, mm_col] = 1
        else:
            df.at[idx, 'MM_Unknown'] = 1
    
    # Handle fuel encoding
    df['fuel'] = df['fuel'].map({'Petrol': 1.0, 'Diesel': 1.1, 'Hybrid': 2.0})
    
    # Handle transmission encoding
    if train_fit:
        ordinal_encoder = OrdinalEncoder(categories=[['Manual', 'Automatic']])
        df[['transmission']] = ordinal_encoder.fit_transform(df[['transmission']])
    else:
        df[['transmission']] = scalers["ordinal_encoder"].transform(df[['transmission']])
    
    # Scale numerical features
    if train_fit:
        scaler_standard = StandardScaler()
        scaler_minmax = MinMaxScaler()
        df[['engine']] = scaler_standard.fit_transform(df[['engine']])
        df[['mileage', 'age']] = scaler_minmax.fit_transform(df[['mileage', 'age']])
    else:
        df[['engine']] = scalers["scaler_standard"].transform(df[['engine']])
        df[['mileage', 'age']] = scalers["scaler_minmax"].transform(df[['mileage', 'age']])
    
    # Select and order features exactly as model expects (excluding price)
    expected_features = [
        'engine', 'transmission', 'fuel', 'mileage', 'age', 'assembly_encoded'
    ] + expected_mm_columns
    
    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[expected_features]

# Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Test database connection
        conn = get_db()
        conn.execute("SELECT 1 FROM car_data LIMIT 1")
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
