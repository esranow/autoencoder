import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import io

# --- MODEL ARCHITECTURE (UNTOUCHED) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 14), nn.ReLU(), nn.Linear(14, 7), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(7, 14), nn.ReLU(), nn.Linear(14, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

app = FastAPI(title="Fraud Detection API")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder()
ANOMALY_THRESHOLD = 1.0 # From 95th percentile logic

@app.on_event("startup")
def load_model():
    try:
        model.load_state_dict(torch.load("model/fraud_autoencoder.pth", map_location=device, weights_only=True))
        model.eval().to(device)
    except Exception as e:
        print(f"Warning: {e}")

class PredictionRequest(BaseModel):
    features: List[float]

# --- UI LAYER (MONOLITHIC HTML) ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Scanner Portal</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; background: #f9f9f9; }
            .card { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            textarea, input[type="file"] { width: 100%; padding: 10px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            button { background: #000; color: #fff; padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; font-weight: bold; }
            button:hover { background: #333; }
            .alert { padding: 15px; margin-top: 15px; border-radius: 4px; font-weight: bold; display: none; }
            .fraud { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
            .normal { background: #dcfce7; color: #166534; border: 1px solid #4ade80; }
            h2, h3 { margin-top: 0; }
        </style>
    </head>
    <body>
        <h2>Internal Transaction Scanner</h2>
        
        <div class="card">
            <h3>Quick Test (Single Transaction)</h3>
            <form onsubmit="checkFraud(event)">
                <label>Raw Transaction Vector (30 comma-separated values):</label><br><br>
                <textarea id="features" rows="4" placeholder="0.0, -1.3, 2.4..."></textarea>
                <button type="submit">Scan Transaction</button>
            </form>
            <div id="single-result" class="alert"></div>
        </div>

        <div class="card">
            <h3>Batch Processing (CSV Upload)</h3>
            <p style="font-size: 0.9em; color: #666;">Upload a CSV containing transactions. The system will append 'Reconstruction_MSE' and 'T_F' columns and return the file.</p>
            <form onsubmit="processBatch(event)">
                <input type="file" id="csv-file" accept=".csv" required>
                <button type="submit" id="batch-btn">Process & Download</button>
            </form>
            <div id="batch-result" class="alert" style="background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc;"></div>
        </div>

        <script>
            // Single Prediction Logic
            async function checkFraud(e) {
                e.preventDefault();
                const resDiv = document.getElementById('single-result');
                const rawData = document.getElementById('features').value;
                const f = rawData.split(',').map(Number);
                
                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({features: f})
                    });
                    
                    if (!res.ok) throw new Error("Invalid input.");
                    const data = await res.json();
                    
                    resDiv.style.display = 'block';
                    resDiv.className = data.is_anomaly ? 'alert fraud' : 'alert normal';
                    resDiv.innerText = data.is_anomaly 
                        ? `⚠️ HIGH RISK DETECTED (MSE: ${data.mse.toFixed(4)})` 
                        : `✅ CLEAR (MSE: ${data.mse.toFixed(4)})`;
                } catch (err) {
                    resDiv.style.display = 'block';
                    resDiv.className = 'alert fraud';
                    resDiv.innerText = "Error. Ensure exactly 30 numeric values.";
                }
            }

            // Batch Prediction Logic
            async function processBatch(e) {
                e.preventDefault();
                const fileInput = document.getElementById('csv-file');
                const resDiv = document.getElementById('batch-result');
                const btn = document.getElementById('batch-btn');
                
                if (fileInput.files.length === 0) return;
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                
                btn.innerText = "Processing...";
                btn.disabled = true;
                resDiv.style.display = 'none';

                try {
                    const res = await fetch('/predict_batch', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!res.ok) throw new Error("Failed to process file.");
                    
                    // Trigger file download from blob
                    const blob = await res.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = "scanned_" + fileInput.files[0].name;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    window.URL.revokeObjectURL(url);
                    
                    resDiv.style.display = 'block';
                    resDiv.innerText = "Processing complete. File downloaded.";
                } catch (err) {
                    resDiv.style.display = 'block';
                    resDiv.className = 'alert fraud';
                    resDiv.innerText = "Error processing CSV. Check format.";
                } finally {
                    btn.innerText = "Process & Download";
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

# --- PREDICTION ENDPOINTS ---

@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")
    
    f = list(request.features)
    # Scale Time and Amount (Indices 0 and 29 based on original dataframe)
    f[0] = (f[0] - 94813.859575) / 47488.145953
    f[29] = (f[29] - 88.349619) / 250.120109
    
    with torch.no_grad():
        t = torch.tensor([f], dtype=torch.float32).to(device)
        out = model(t)
        mse = torch.mean((t - out) ** 2).item()
        
    return {"mse": mse, "is_anomaly": mse > ANOMALY_THRESHOLD}

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Make a copy for scaling so we don't alter the output data values
        df_scaled = df.copy()
        
        # Apply scaling if standard column names exist, otherwise assume positional
        if 'Time' in df_scaled.columns and 'Amount' in df_scaled.columns:
            df_scaled['Time'] = (df_scaled['Time'] - 94813.859575) / 47488.145953
            df_scaled['Amount'] = (df_scaled['Amount'] - 88.349619) / 250.120109
            # Assume V1-V28 exist
            feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            features = df_scaled[feature_cols].values
        else:
            # Fallback: Just grab the first 30 columns and scale 0 and 29
            features = df_scaled.iloc[:, :30].values.astype(float)
            features[:, 0] = (features[:, 0] - 94813.859575) / 47488.145953
            features[:, 29] = (features[:, 29] - 88.349619) / 250.120109

        # Batch prediction
        t_features = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            reconstructions = model(t_features)
            # Calculate MSE for each row across the 30 features
            mses = torch.mean((t_features - reconstructions) ** 2, dim=1).cpu().numpy()
        
        # Append results to the original dataframe
        df['Reconstruction_MSE'] = mses
        df['T_F'] = mses > ANOMALY_THRESHOLD
        
        # Convert back to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            output, 
            media_type="text/csv", 
            headers={"Content-Disposition": f"attachment; filename=scanned_{file.filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")