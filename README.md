

# ECG Classification Project

## Overview
This project implements a **hybrid CNN + LSTM deep learning model** to classify ECG signals for early detection of cardiac abnormalities. It includes a **Streamlit dashboard** for uploading ECG CSV files and viewing predictions.

## Tech Stack
- **Programming & ML:** Python, TensorFlow/Keras, NumPy, Scikit-learn
- **Deep Learning:** CNN, LSTM
- **Dashboard & Visualization:** Streamlit, Matplotlib
- **Data Processing:** StandardScaler, One-hot Encoding

## Project Structure
```
ECG_Classification_Project/
│
├── app.py                # Streamlit dashboard for ECG prediction
├── model_training.py     # Script to train the hybrid CNN+LSTM model
├── ecg_hybrid_model.h5   # Trained model file
├── ecg_data.csv          # Placeholder/sample dataset (user to provide real data)
└── README.md             # Project documentation
```

## Usage

### 1. Train the Model
```bash
python model_training.py
```
- Trains the hybrid CNN + LSTM model on your ECG dataset
- Saves the trained model as `ecg_hybrid_model.h5`

### 2. Run Streamlit Dashboard
```bash
streamlit run app.py
```
- Upload your ECG CSV file
- View predicted classes for each signal

## Dataset Requirements
- CSV format with multiple columns for ECG signals and one column named `label` for the target class.
- Compatible with MIT-BIH Arrhythmia Dataset or similar ECG datasets.

## Extending the Project
- Add **automated diagnostic reports** (PDF/Excel) using model predictions
- Support **real-time ECG streaming**
- Enhance model accuracy with additional CNN/LSTM layers or data augmentation

## License
Educational and research purposes only.
