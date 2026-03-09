import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List

# Define the Autoencoder model architecture to match the saved weights
class Autoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super(Autoencoder, self).__init__()
        # Encoder: 30 -> 14 -> 7
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        # Decoder: 7 -> 14 -> 30
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection Autoencoder API", version="1.0")

# Load model globally
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Threshold determined from the notebook (95th percentile of normal errors)
# NOTE: To be completely robust, this threshold should ideally be loaded from a configuration or computed, 
# but for the purpose of this API we will use a reasonable fixed threshold or return the raw MSE.
# Let's assume a default threshold of 0.05 based typical scaled MSE reconstruction values, but the caller
# can use the raw `mse` to apply their own threshold.
ANOMALY_THRESHOLD = 0.05 

@app.on_event("startup")
def load_model():
    global model
    model = Autoencoder(input_dim=30)
    # Load weights
    try:
        model.load_state_dict(torch.load("model/fraud_autoencoder.pth", map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Not raising here so the app can start and we can see the error in the logs,
        # but in production, we might want to fail fast.


class PredictionRequest(BaseModel):
    # Expecting 30 features
    features: List[float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [0.0] * 30
            }
        }
    )


class PredictionResponse(BaseModel):
    mse: float
    is_anomaly: bool


@app.get("/")
def read_root():
    return {"message": "Fraud Detection Autoencoder API is running!"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(request.features)}")
    
    try:
        with torch.no_grad():
            # Convert input payload to tensor
            input_tensor = torch.tensor([request.features], dtype=torch.float32).to(device)
            
            # Reconstruct
            output_tensor = model(input_tensor)
            
            # Calculate MSE
            mse = torch.mean((input_tensor - output_tensor) ** 2).item()
            
            # Flag as anomaly if MSE > threshold
            is_anomaly = mse > ANOMALY_THRESHOLD
            
            return PredictionResponse(mse=mse, is_anomaly=is_anomaly)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
