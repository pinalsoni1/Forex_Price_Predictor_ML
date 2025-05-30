📈 Forex Price Prediction & Direction ML Project

A machine learning project focused on predicting the mid-price and classifying the direction of Forex (foreign exchange) tick data using large-scale historical datasets. This project leverages Random Forest Regressor for continuous price prediction and XGBoost Classifier for directional classification (up/down), making it suitable for algorithmic trading or financial forecasting.

⸻

🔍 Project Objectives
	•	📊 Mid-Price Forecasting: Predict the future mid-price of a currency pair using historical bid-ask data.
	•	🔺 Price Direction Classification: Predict whether the next price movement will be upward (1) or downward/no change (0).

⸻

🧠 Machine Learning Models

✅ Random Forest Regressor (for Mid-Price Prediction)
	•	Predicts future mid-prices using ensemble decision trees.
	•	Handles noise and outliers in financial data well.
	•	Optimized using hyperparameter tuning and cross-validation.
	•	Pipeline steps:
	•	Feature vectorization (VectorAssembler)
	•	Feature scaling (StandardScaler)
	•	Model training (RandomForestRegressor)

✅ XGBoost Classifier (for Price Direction Classification)
	•	Gradient boosting decision tree algorithm ideal for imbalanced or noisy datasets.
	•	Learns complex patterns in tick-level financial time series data.
	•	Achieved directional prediction accuracy of up to 71.69% on 8 million tick samples.

⸻

🗃️ Features Used

🔹 For Regression (Price Prediction)
	•	Mid Price = (Bid + Ask) / 2
	•	Mid Lag 1 = Previous mid-price
	•	Mid Return = % change in mid-price
	•	Mid MA5 = 5-point moving average of mid-price

🔹 For Classification (Price Direction)
	•	Spread = Ask - Bid
	•	Volume Imbalance = Normalized buyer/seller dominance
	•	Next Ask Price (used to derive the label, not as a feature)
	•	Label = 1 (Ask price increased) or 0 (Ask same/decreased)

⸻

🧪 Model Performance

📉 Random Forest Regressor (Price Prediction)
	•	Achieved low RMSE (Root Mean Squared Error) values across various data partitions
	•	Robust to noise and outliers, making it suitable for volatile financial data
	•	Performance remained stable as dataset scaled from 2M to 10M rows
	•	Showed strong parallel scalability — training time dropped significantly with more Spark worker nodes

📈 XGBoost Classifier (Price Direction)
	•	Achieved classification accuracy up to 71.69% on 8M tick records
	•	Accuracy improved steadily from 2M to 8M rows, then slightly decreased at 10M due to possible overfitting or noise
	•	Effectively captured nonlinear patterns in price movement
	•	Tuned using cross-validation for optimal hyperparameters

⸻

⚙️ Technologies Used
	•	Python (PySpark, Scikit-learn, XGBoost)
	•	Apache Spark (for big data processing)
	•	Jupyter Notebooks / Databricks
	•	Git & GitHub

⸻

🚀 Scalability Testing
	•	Scale-Up: Increasing data size led to linear growth in training time (11.7 to 24.9 minutes from 2M to 10M records).
	•	Scale-Out: Training time significantly reduced when using more Spark worker nodes.


⸻

⚠️ Note on Data

Due to GitHub’s 100MB limit, large tick datasets (4M–10M rows) are not included in this repo.
To replicate:
	•	Use your own Forex tick data (e.g., from Dukascopy)
	•	Or enable Git LFS for large file handling
