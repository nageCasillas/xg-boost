# XGBoost Models

This repository explores both Regression and Classification models using the XGBoost algorithm. It features a Streamlit interface that allows users to switch between models and input parameters for predictions.

## About
This project demonstrates the capabilities of the XGBoost algorithm by implementing:

- **Regression**: Predicting used car prices based on features from the [CarDekho Used Car Dataset](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data).
- **Classification**: Predicting holiday package purchase decisions based on features from the [Holiday Package Purchase Prediction Dataset](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction).

The interactive Streamlit app enables users to seamlessly switch between regression and classification models and input their data for real-time predictions.

## Features

- **Model Selection**: Choose between regression (Car Price Prediction) and classification (Holiday Package Purchase Prediction) models.
- **Parameter Input**: Provide feature values directly in the UI for live predictions.
- **Interactive Interface**: Real-time predictions and performance visualizations for selected models.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.x
- pip package manager

### Installation

Clone the repository:

```bash
git clone https://github.com/nageCasillas/xg-boost.git
cd xg-boost
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and go to [http://localhost:8501](http://localhost:8501) to start interacting with the app.

## Datasets

- **CarDekho Used Car Dataset (Regression)**: [Download here](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data).
- **Holiday Package Purchase Dataset (Classification)**: [Download here](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction).

## Usage

### Regression: Car Price Prediction
Input features such as car name, vehicle age, kilometers driven, fuel type, and more to predict the selling price of a car.

### Classification: Holiday Package Purchase Prediction
Provide customer-related features such as age, occupation, monthly income, and number of trips to predict whether a customer is likely to purchase a holiday package.

## Contribution

Contributions are welcome! If you would like to contribute, please:

1. Fork this repository.
2. Create a new branch (`feature-branch-name`).
3. Make changes and test them.
4. Create a pull request.

