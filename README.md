# 🏃 Activity Predictor & HAR Studio Pro

An end-to-end Machine Learning project to recognize human movement using smartphone sensors.

## 🌟 What This Project Does
This studio takes complex signal data from accelerometers and gyroscopes to automatically identify if a person is standing, walking, or laying down. It turns "unreadable" sensor data into a visual intelligence dashboard using K-Means clustering.

## 🛠️ Tech Stack
- **Languages**: Python
- **AI Libraries**: Scikit-Learn (K-Means, PCA, StandardScaler)
- **Interface**: Streamlit
- **Visualization**: Matplotlib, Seaborn
- **Data**: Pandas, NumPy

## 📂 Project Structure
- `app.py` → Streamlit interactive UI
- `analysis.py` → Model training script
- `models/` → Trained ML models (.pkl files)
- `data/` → Contains `train.csv` and `test.csv` (Requires manual download/placement)
- `utils.py` → Helper functions

---

## 🚀 Quick Start (Windows)

Follow these exact steps to run the project on a Windows machine:

1. **Open your Terminal (Command Prompt or PowerShell)** and navigate to the project folder.

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - Command Prompt:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - PowerShell:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

4. **Install all required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the Data & Models**:
   - Ensure your dataset (`train.csv` and `test.csv`) is placed inside a folder named `data`.
   - Run the analysis script to generate the Machine Learning models:
   ```bash
   python analysis.py
   ```

6. **Start the Web App**:
   ```bash
   streamlit run app.py
   ```
   A browser window will open automatically with the beautiful new UI.

---
**Simple. Professional. Intelligent.**
