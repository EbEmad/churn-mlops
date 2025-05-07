from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np



# load model
model=joblib.load('../models/best_model.pkl')
scaler=joblib.load('../models/scaler.pkl')
feature_columns=joblib.load('../models/columns.pkl')


app=FastAPI(title='Telco Churn Predictor API')

# input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str  # Expected to be one of: ['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def preprocess_input(data: dict):
    df=pd.DataFrame([data])

    # Manual encoding to match training
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce').fillna(0)

    binary_cols=['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for col in binary_cols:
        df[col]=df[col].map({'Yes':1 ,'No':0})
    
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})


    # one hot encoding

    df=pd.get_dummies(df,columns=[
              'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ])


    # Ensure same column order as traing

    for col in feature_columns:
        if col not in df.columns:
            df[col]=0
    df=df[feature_columns]

    # scale
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]=scaler.transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    return df

@app.get('/')
def root():
    return {"message": "Telco Churn Predictor API is live 🎯"}

@app.post('/predict')
def predict(data:CustomerData):
    input_df=preprocess_input(data.dict())
    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0][1]

    return {
        'prediction': int(prediction),
        'churn_probability': round(float(probability),4)
    }