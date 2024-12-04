# SVD Recommendation System

This project implements a **Singular Value Decomposition (SVD)** model to analyze user-item interactions. It is trained on a dataset containing user ratings, evaluates model performance using metrics such as Precision, Recall, F1-Score, and NDCG, and offers optional evaluation features like fairness, transparency, controllability, and robustness.

## Features

- **Data Loading & Preprocessing**:

  - Load user-item interaction data from a JSON file.
  - Preprocess data into numerical user and item IDs along with their ratings.

- **SVD Model**:

  - Initialize latent factors for users and items.
  - Train the model using gradient descent to optimize predictions.

- **Evaluation Metrics**:

  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
  - Precision, Recall, F1-Score, and NDCG for Top-N Recommendations.

- **Fairness, Transparency & Controllability**:

  - Evaluate fairness across user groups.
  - Explain recommendations based on feature contributions.
  - Group users into clusters for better personalization.

- **Privacy & Robustness**:
  - Anonymize user IDs in recommendations.
  - Simulate adversarial attacks on model predictions to test robustness.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/svd-recommendation-system.git
   cd svd-recommendation-system
   ```
