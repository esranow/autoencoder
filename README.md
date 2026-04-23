# 🕵️ Fraud Detection Autoencoder

This repository contains an end-to-end Machine Learning pipeline and web portal using a **Deep Learning Autoencoder** designed specifically to detect fraudulent financial transactions. It provides a robust, fast, and scalable solution packed with an interactive Streamlit UI and a production-ready FastAPI backend.

## 🧠 Why an Autoencoder Approach?

For financial fraud detection, traditional classification models often struggle because fraudulent transactions make up a microscopic fraction of overall data (extreme class imbalance). 

**We take a different approach:**
1. **Unsupervised Anomaly Detection:** The Autoencoder is trained primarily on *normal* transactions to learn their latent representation. It acts as a compression and decompression algorithm for expected behaviors.
2. **Reconstruction Error as a Fraud Signal:** When the Autoencoder attempts to process a fraudulent transaction, it fails to reconstruct the data accurately because it hasn't learned the patterns of fraud. This results in a high **Mean Squared Error (MSE)**.
3. **Thresholding:** By setting an anomaly threshold (typically the 95th+ percentile of normal reconstruction errors), any transaction exceeding this MSE is instantly flagged as **High Risk**.

This eliminates the need for complex, biased data-balancing strategies like SMOTE and forces the model to intuitively single out behavior that just "doesn't look right".

## ✨ What Makes This Project So Better

1. **Dual-Interface System**:
   * **Streamlit Web Portal:** An elegant, highly visual dashboard designed for analysts to quickly check single transactions or upload bulk CSVs without any coding knowledge.
   * **FastAPI Backend:** A high-concurrency, developer-friendly REST API for programmatic integrations, featuring JSON and Multipart Form routes.
2. **Native Preprocessing Integration**: The system handles the standard scaling of the `Time` and `Amount` fields dynamically. You can input raw values straight from your database, and the application will correctly scale them before passing them to the PyTorch instance.
3. **Optimized PyTorch Model**: Lightweight `nn.Sequential` feed-forward architecture ensures blazing fast inference times, allowing for real-time transaction scanning.
4. **Batch Processing Capabilities**: Users can upload massive CSV files. The model efficiently uses PyTorch vectorization to process the batch, calculate the Reconstruction MSE, identify fraud, and dynamically return the appended CSV instantly.

## 🚀 Getting Started

### 1. Requirements

Ensure you have Python installed, then run:
```bash
pip install torch fastapi uvicorn pandas streamlit
```

### 2. Running the Interactive UI (Streamlit)

To launch the web visualization dashboard:
```bash
streamlit run streamlit_app.py
```
*Access the portal at `http://localhost:8501`*

### 3. Running the REST API (FastAPI)

To launch the API endpoints for system integrations:
```bash
uvicorn main:app --reload
```
*Access interactive API docs at `http://localhost:8000/docs`*

## 📁 Repository Structure

- `main.py` - Core FastAPI web application and PyTorch Autoencoder definition.
- `streamlit_app.py` - Interactive Streamlit web-based UI.
- `model/` - Contains the saved `.pth` PyTorch model weights.
- `creditcard.csv` - The local dataset.
- `Dockerfile` & `docker-compose.yml` - Containerization for easy cloud deployment.
