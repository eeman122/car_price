from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import mlflow
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder

# Configuration
DB_URL = "postgresql://postgres:1234@localhost:5432/car_db"
MLFLOW_TRACKING_URI = "http://localhost:5000"
DATA_DIR = Path("data")
MODEL_DIR = Path("models")

@task(name="Extract Data from CSV")
def extract_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    logger = get_run_logger()
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

@task(name="Load Data to PostgreSQL")
def load_to_postgres(df: pd.DataFrame, table_name: str):
    """Load data into PostgreSQL database"""
    logger = get_run_logger()
    logger.info(f"Loading data to PostgreSQL table {table_name}")
    
    engine = create_engine(DB_URL)
    try:
        df.to_sql(
            table_name, 
            engine, 
            if_exists='replace', 
            index=False,
            method='multi',
            chunksize=1000
        )
        logger.info(f"Successfully loaded {len(df)} records to {table_name}")
    except Exception as e:
        logger.error(f"Error loading to PostgreSQL: {e}")
        raise
    finally:
        engine.dispose()

@task(name="Preprocess Data")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw data"""
    logger = get_run_logger()
    logger.info("Starting data preprocessing")
    
    try:
        # Handle missing values
        df = df.dropna(subset=['price', 'year', 'engine'])
        
        # Calculate age
        current_year = datetime.now().year
        df['age'] = current_year - df['year']
        
        # Create make_model feature
        df['make_model'] = df['make'] + '_' + df['model']
        
        # Fill missing categorical values
        df['assembly'] = df['assembly'].fillna('Local')
        df['transmission'] = df['transmission'].fillna('Manual')
        df['fuel'] = df['fuel'].fillna('Petrol')
        
        logger.info("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

@task(name="Train and Log Model")
def train_model(X: pd.DataFrame, y: pd.Series):
    """Train model and log with MLFlow"""
    logger = get_run_logger()
    logger.info("Starting model training")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Car Price Prediction")
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and log linear model
        with mlflow.start_run(run_name="Linear Regression"):
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred = linear_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metrics({
                "rmse": rmse,
                "r2_score": r2
            })
            
            mlflow.sklearn.log_model(
                linear_model,
                "linear_model",
                registered_model_name="car_price_linear"
            )
            
            logger.info(f"Linear model trained - RMSE: {rmse:.2f}, R2: {r2:.2f}")
        
        # Train and log polynomial model
        with mlflow.start_run(run_name="Polynomial Regression"):
            poly_features = PolynomialFeatures(degree=2)
            poly_model = Pipeline([
                ('poly', poly_features),
                ('linear', LinearRegression())
            ])
            poly_model.fit(X_train, y_train)
            y_pred = poly_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metrics({
                "rmse": rmse,
                "r2_score": r2
            })
            
            mlflow.sklearn.log_model(
                poly_model,
                "poly_model",
                registered_model_name="car_price_poly"
            )
            
            logger.info(f"Polynomial model trained - RMSE: {rmse:.2f}, R2: {r2:.2f}")
            
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

@task(name="Prepare Features")
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for modeling"""
    logger = get_run_logger()
    logger.info("Preparing features for modeling")
    
    try:
        # Feature selection
        features = df[[
            'make_model', 'year', 'engine', 'mileage', 'transmission', 
            'fuel', 'assembly', 'age'
        ]].copy()
        
        # Target variable
        target = df['price']
        
        # Create mappings for categorical features
        top_make_models = features['make_model'].value_counts().nlargest(20).index
        features['make_model'] = features['make_model'].apply(
            lambda x: x if x in top_make_models else 'Other'
        )
        
        # One-hot encoding
        features = pd.get_dummies(
            features, 
            columns=['make_model', 'transmission', 'fuel', 'assembly'],
            drop_first=True
        )
        
        # Save scalers for future use
        scalers = {}
        scalers['scaler_standard'] = StandardScaler().fit(features[['engine']])
        scalers['scaler_minmax'] = MinMaxScaler().fit(features[['mileage', 'age']])
        scalers['ordinal_encoder'] = OrdinalEncoder(
            categories=[['Manual', 'Automatic']]
        ).fit(features[['transmission_Manual']])
        
        joblib.dump(scalers, MODEL_DIR / "scalers.pkl")
        
        return features, target
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise

@flow(name="Car Price Prediction Pipeline", task_runner=SequentialTaskRunner())
def car_price_pipeline(csv_file_path: str):
    """Main pipeline for car price prediction system"""
    # Extract data
    raw_data = extract_data(csv_file_path)
    
    # Load to PostgreSQL
    load_to_postgres(raw_data, "raw_car_data")
    
    # Preprocess data
    cleaned_data = preprocess_data(raw_data)
    load_to_postgres(cleaned_data, "cleaned_car_data")
    
    # Prepare features
    features, target = prepare_features(cleaned_data)
    
    # Train models
    train_model(features, target)
    
    logger = get_run_logger()
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    # Example usage
    car_price_pipeline("data/car_data.csv")