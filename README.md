# 📈 Stock Price Prediction using LSTM (Time Series Forecasting)

A deep learning-based stock price prediction system that forecasts future stock closing prices using a Long Short-Term Memory (LSTM) neural network. The project leverages historical stock market data to perform time series forecasting and provides an interactive Streamlit web application for visualization and prediction.

---

## 📌 Project Overview

Stock price forecasting is one of the most challenging tasks in financial analytics due to the dynamic and volatile nature of stock markets. This project utilizes an LSTM (Long Short-Term Memory) network, a specialized Recurrent Neural Network (RNN), to learn temporal dependencies from historical stock prices and predict future closing prices.

The application allows users to:

- View historical stock prices.
- Select a prediction date.
- Predict the closing price using the trained LSTM model.
- Compare actual vs predicted prices.
- Evaluate model performance using standard regression metrics.

---

## 🚀 Features

- Historical Stock Price Visualization
- LSTM-based Time Series Forecasting
- Date-wise Prediction
- Interactive Streamlit Web Application
- Actual vs Predicted Price Comparison
- Performance Metrics (MAE & RMSE)
- Responsive User Interface

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning & Deep Learning
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

### Data Visualization
- Matplotlib

### Web Framework
- Streamlit

---

## 📂 Project Structure

```
Stock-Price-Prediction/
│
├── app.py                     # Streamlit Application
├── yahoo_stock.csv            # Stock Dataset
├── stock_lstm_model.keras     # Trained LSTM Model
├── scaler.pkl                 # Saved MinMaxScaler
├── Stock_Prediction.ipynb     # Model Training Notebook
├── requirements.txt           # Required Libraries
├── README.md                  # Project Documentation
```

---

## 📊 Dataset

The dataset contains historical stock market data with the following features:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

Target Variable:

- Close Price

---

## 🧠 Model Architecture

The model is built using Long Short-Term Memory (LSTM) networks for time series forecasting.

Architecture:

- Input Layer
- LSTM Layer
- Dropout Layer
- Dense Layer
- Output Layer

Input Sequence Length:

```
60 previous trading days
```

Output:

```
Next Day Closing Price
```

---

## 📈 Model Evaluation

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The Streamlit application also visualizes:

- Historical Closing Price
- Actual vs Predicted Price
- Prediction Comparison Table

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Stock-Price-Prediction.git
```

Move into project directory

```bash
cd Stock-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

The application includes:

- Historical Price Chart
- Dataset Preview
- Date-wise Stock Prediction
- Prediction Metrics
- Actual vs Predicted Graph

---

## 📚 Future Improvements

- Multi-day Forecasting (7-Day & 30-Day)
- Real-time Stock Data Integration
- Sentiment Analysis using Financial News
- Technical Indicators (RSI, MACD, EMA)
- Multiple Stock Selection
- Model Comparison (LSTM, GRU, Bi-LSTM)

---

## 🎯 Learning Outcomes

- Time Series Forecasting
- Deep Learning using LSTM
- Data Preprocessing
- Model Evaluation
- Streamlit Deployment
- Financial Data Analysis

---

## 📄 License

This project is developed for educational and research purposes.

---

## 👨‍💻 Author

**Avneesh Chaurasia**

AI & Machine Learning Engineering Student

Interested in:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Data Science
- Time Series Forecasting

---
⭐ If you found this project useful, consider giving it a star on GitHub!
