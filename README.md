# AtRisk Dashboard

A Streamlit-based dashboard application for analyzing and predicting company risk factors using machine learning models.

## Overview

This project provides a web-based dashboard that:
- Analyzes company data to assess risk factors
- Uses machine learning models to predict risk probabilities
- Visualizes data and model insights
- Provides interactive features for data exploration

## Features

- Interactive data visualization
- Risk probability predictions
- Model performance metrics
- Data export capabilities
- Real-time analysis

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/zhamgg/atrisk_dashboard.git
cd atrisk_dashboard
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the dashboard interface to:
   - Upload and analyze company data
   - View risk predictions
   - Explore model insights
   - Export results

## Project Structure

- `app.py`: Main Streamlit application
- `model_builder.py`: Machine learning model implementation
- `demo_data.py`: Sample data generation and processing
- `requirements.txt`: Project dependencies

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.2
- joblib >= 1.0.1
- imbalanced-learn >= 0.8.0
- matplotlib >= 3.4.0
- xgboost >= 1.7.0
- streamlit >= 1.22.0
- seaborn >= 0.11.2

## License

Proprietary