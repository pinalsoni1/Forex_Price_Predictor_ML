📈 Forex Price Prediction & Direction Modeling at Scale

Overview

This project explores whether short-term price movement in high-frequency Forex markets can be predicted using historical tick-level data, and whether such models can be trained efficiently at scale.

High-frequency financial data is extremely noisy and fast-changing. Individual price movements are often random, which makes naive prediction unreliable.
The goal of this project is not perfect accuracy, but to extract stable, short-term signals and evaluate whether they can be learned, validated, and retrained in a scalable way.

Two complementary modeling tasks are explored:
	•	Price prediction (regression): What is the next price likely to be?
	•	Price direction (classification): Is the price more likely to move up or down next?

⸻

Project Objectives
	•	Mid-Price Forecasting
Predict the next mid-price using recent bid-ask history.
	•	Price Direction Classification
Predict whether the next price movement is upward or downward.
	•	Scalability & Feasibility
Evaluate whether the modeling approach remains practical as data volume and compute scale increase.

⸻
How to Run This Project

This project is designed for a distributed Spark environment (Databricks recommended).

Requirements
	•	Python
	•	PySpark
	•	XGBoost
	•	scikit-learn

Steps
	1.	Set up a Spark environment (I have used Azure Databricks)
	2.	Load Forex tick data with required columns
	3.	Run notebooks/scripts in order:
	•	Data preprocessing
	•	Feature engineering
	•	Model training and validation
	•	Scalability experiments

⚠️ Note: Large datasets (4M–10M rows) are not included due to GitHub size limits.

⸻

Data Overview

The dataset consists of high-frequency Forex tick data with millions of records, including:
	•	UTC timestamps
	•	Bid and ask prices
	•	Bid and ask volumes

At the tick level, prices fluctuate constantly and contain substantial noise.
A key challenge of this project is transforming raw, noisy observations into time-aware features that capture short-term market behavior without leaking future information.

⸻

Feature Engineering

Feature engineering was designed to be time-aware, using only past information to reflect real-world deployment.

Price Prediction (Regression)
	•	Mid Price = (Bid + Ask) / 2
	•	Lagged Mid Price (previous value)
	•	Mid Return (% change)
	•	Mid MA5 (short moving average)

These features capture price level, momentum, and short-term trend while smoothing noise.

Price Direction (Classification)
	•	Spread = Ask − Bid (liquidity / uncertainty)
	•	Volume Imbalance (buy vs sell pressure)
	•	Label derived from next-step price movement
(Used only to define the outcome, not as a feature)

⸻

Modeling Approach

Models Used

Random Forest Regressor (Price Prediction)
	•	Ensemble of decision trees for continuous price estimation
	•	Robust to noise and outliers
	•	Trained using a Spark ML pipeline:
	•	VectorAssembler
	•	StandardScaler
	•	RandomForestRegressor

XGBoost Classifier (Price Direction)
	•	Gradient boosting decision trees
	•	Well-suited for noisy, non-linear patterns
	•	Achieved directional accuracy of up to ~72% on large datasets

⸻

Validation Strategy
	•	Chronological train–test split
Data is split by time to ensure the model only learns from past observations and is evaluated on future data.
	•	2-fold cross-validation (training set only)
Used for hyperparameter tuning to balance stability and computational cost.

This approach provides more reliable model comparison than a single split while remaining practical for large-scale data.

⸻

Evaluation Metrics

Price Prediction
	•	RMSE (Root Mean Squared Error)
Chosen because it penalizes large prediction errors more heavily, which is important in short-term financial forecasting.

Price Direction
	•	Accuracy – overall correctness
	•	Precision – reliability of upward signals
	•	Recall – ability to capture true upward movements

⸻

Scalability Experiments

To evaluate feasibility at scale, both scale-up and scale-out experiments were conducted.
	•	Scale-Up:
Increasing data size led to predictable, near-linear growth in training time.
	•	Scale-Out:
Adding Spark worker nodes reduced training time by nearly 3×, without changing model accuracy.

Key insight:
Scaling primarily improves training speed and feasibility, not predictive performance.

⸻

Key Design Choices & Rationale

	•	Separate regression and classification tasks to match distinct problem goals
	•	Time-aware features (lags, rolling windows) to avoid data leakage
	•	RMSE chosen to penalize large price errors
	•	Precision–recall tradeoff explicitly analyzed for direction prediction
	•	Scalability evaluated to ensure models can be retrained realistically

⸻

Learnings
	•	Problem framing and metric choice matter more than model complexity in noisy, high-frequency data
	•	Time-aware feature design is critical for extracting meaningful signals
	•	At very short horizons, sensitivity is more realistic than perfect accuracy
	•	Scaling affects feasibility and iteration speed, not model accuracy

⸻

Next Steps
	•	Move to fully time-based validation (rolling or expanding windows)
	•	Explore longer prediction horizons
	•	Tune decision thresholds to balance precision vs recall
	•	Add automated retraining and performance monitoring

⸻

Technologies Used
	•	Python (PySpark, scikit-learn, XGBoost)
	•	Apache Spark / Databricks
	•	Jupyter Notebooks
	•	Git & GitHub
