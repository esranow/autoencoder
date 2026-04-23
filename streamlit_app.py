import streamlit as st
import torch
import pandas as pd
import io

# Import the model architecture and configuration from main.py
from main import Autoencoder, ANOMALY_THRESHOLD, device

# Make sure model is evaluated
try:
    model = Autoencoder()
    model.load_state_dict(torch.load("model/fraud_autoencoder.pth", map_location=device, weights_only=True))
    model.eval().to(device)
except Exception as e:
    st.warning(f"Note: model weights may not be loaded properly: {e}")

st.set_page_config(page_title="Fraud Scanner Portal", page_icon="🕵️", layout="centered")
st.title("💳 Internal Transaction Scanner")
st.markdown("---")

st.header("⚡ Quick Test (Single Transaction)")
raw_data = st.text_area("Raw Transaction Vector (30 comma-separated values):", placeholder="0.0, -1.3, 2.4...")
if st.button("Scan Transaction"):
    if not raw_data:
        st.warning("Please enter transaction data.")
    else:
        try:
            f = [float(x.strip()) for x in raw_data.split(',')]
            if len(f) != 30:
                st.error(f"Expected 30 features, got {len(f)}")
            else:
                # Scale Time and Amount
                f[0] = (f[0] - 94813.859575) / 47488.145953
                f[29] = (f[29] - 88.349619) / 250.120109
                
                with torch.no_grad():
                    t = torch.tensor([f], dtype=torch.float32).to(device)
                    out = model(t)
                    mse = torch.mean((t - out) ** 2).item()
                
                if mse > ANOMALY_THRESHOLD:
                    st.error(f"⚠️ **HIGH RISK DETECTED** (MSE: {mse:.4f})")
                else:
                    st.success(f"✅ **CLEAR** (MSE: {mse:.4f})")
        except Exception as e:
            st.error(f"Error processing input: {e}")

st.markdown("---")
st.header("📂 Batch Processing (CSV Upload)")
st.write("Upload a CSV containing transactions. The system will append `Reconstruction_MSE` and `Is_Fraud` columns.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    if st.button("Process CSV Data"):
        try:
            df = pd.read_csv(uploaded_file)
            df_scaled = df.copy()
            
            if 'Time' in df_scaled.columns and 'Amount' in df_scaled.columns:
                df_scaled['Time'] = (df_scaled['Time'] - 94813.859575) / 47488.145953
                df_scaled['Amount'] = (df_scaled['Amount'] - 88.349619) / 250.120109
                feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                features = df_scaled[feature_cols].values
            else:
                features = df_scaled.iloc[:, :30].values.astype(float)
                features[:, 0] = (features[:, 0] - 94813.859575) / 47488.145953
                features[:, 29] = (features[:, 29] - 88.349619) / 250.120109

            t_features = torch.tensor(features, dtype=torch.float32).to(device)
            with torch.no_grad():
                reconstructions = model(t_features)
                mses = torch.mean((t_features - reconstructions) ** 2, dim=1).cpu().numpy()
            
            df['Reconstruction_MSE'] = mses
            df['Is_Fraud'] = mses > ANOMALY_THRESHOLD
            
            st.success("✅ Processing complete. You can preview and download the results below.")
            
            # Show preview
            st.dataframe(df.head(10))
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Processed CSV",
                data=csv_buffer.getvalue(),
                file_name=f"scanned_{uploaded_file.name}",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
