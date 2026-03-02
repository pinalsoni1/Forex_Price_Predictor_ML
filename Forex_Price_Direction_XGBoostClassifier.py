# Databricks notebook source
# MAGIC %pip install xgboost==2.0.3 pyspark

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, when
from pyspark.sql.window import Window

from xgboost.spark import SparkXGBClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import time
import matplotlib.pyplot as plt
import numpy as np


# COMMAND ----------

# Step 1: Start Spark Session
spark = SparkSession.builder \
    .appName("ForexDirectionPredictionSparkXGB") \
    .getOrCreate()

# COMMAND ----------

# Step 2: Load Data
df = spark.read.csv("dbfs:/FileStore/shared_uploads/psoni4@stevens.edu/chunk_10M.csv", header=True, inferSchema=True)
df.show()

# COMMAND ----------

# Step 3: Feature Engineering
df = df.withColumn("Spread", col("AskPrice") - col("BidPrice"))
df = df.withColumn("VolumeImbalance", 
                   (col("AskVolume") - col("BidVolume")) / (col("AskVolume") + col("BidVolume")))

window = Window.orderBy("UTC")
df = df.withColumn("NextAskPrice", lead("AskPrice", 1).over(window))
df = df.withColumn("Label", when(col("NextAskPrice") > col("AskPrice"), 1).otherwise(0))

feature_cols = ["AskPrice", "BidPrice", "AskVolume", "BidVolume", "Spread", "VolumeImbalance"]
df = df.select(*feature_cols, "Label").na.drop()

# COMMAND ----------

# Step 4: Assemble Features
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# COMMAND ----------

# Step 5: Train-Test Chronological Split

#train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

split_ts = df.approxQuantile("UTC_timestamp", [0.8], 0.0)[0]
train_df = df.filter(df.UTC_timestamp <= split_ts)
test_df  = df.filter(df.UTC_timestamp > split_ts)

# COMMAND ----------

# Step 6: Define XGBoost Classifier
xgb_classifier = SparkXGBClassifier(
    features_col="features",
    label_col="Label",
    prediction_col="prediction",
    use_gpu=False,
    n_workers=2  # number of workers
)

# COMMAND ----------

# Step 7: Build ML Pipeline
pipeline = Pipeline(stages=[assembler, xgb_classifier.copy({})])

# COMMAND ----------

# Step 8: Set up Hyperparameter Grid
param_grid = ParamGridBuilder() \
    .addGrid(xgb_classifier.max_depth, [8, 10]) \
    .build()

# COMMAND ----------

# Step 9: CrossValidator Setup
evaluator = MulticlassClassificationEvaluator(
    labelCol="Label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,  # 3-Fold Cross Validation
    seed=42
)

# COMMAND ----------

# Step 10: Train


start = time.time()
cvModel = cv.fit(train_df)
end = time.time()

print(f"Elapsed time to train the xgboost for price direction: {end-start:.4f} seconds")

# COMMAND ----------

# Step 11: Evaluate on Test Set
predictions = cvModel.transform(test_df)
predictions.select("label", "prediction").show()

accuracy = evaluator.evaluate(predictions)
print(f"Best Model Test Accuracy = {accuracy:.4f}")

# COMMAND ----------

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
precision = precision_evaluator.evaluate(predictions)

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="recallByLabel")
recall = recall_evaluator.evaluate(predictions)

print(f"Precision = {precision:.4f}")
print(f"Recall = {recall:.4f}")

# COMMAND ----------

# Step 12: Get Best Model Params
best_model = cvModel.bestModel.stages[-1]  # Get the XGB stage
print("Best hyperparameters found:")
print(f"max_depth: {best_model.getOrDefault('max_depth')}")
print(f"learning_rate: {best_model.getOrDefault('learning_rate')}")

# COMMAND ----------

# Dataset sizes in millions
data_sizes = [20, 40, 60, 80, 100]

# Training times (in seconds) for each worker configuration
training_times_1w = [156.2885, 364.9474, 531.6567, 717.8642, 888.8057]
training_times_2w = [162.4202, 350.3280, 509.2504, 672.5889, 814.2559]
training_times_3w = [102.2885, 320.7151, 491.5009, 650.4378, 796.1828]

training_times_1w = [round((time / 60), 2) for time in training_times_1w]
training_times_2w = [round((time / 60), 2) for time in training_times_2w]
training_times_3w = [round((time / 60), 2) for time in training_times_3w]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, training_times_1w, marker='o', label='1 Worker')
plt.plot(data_sizes, training_times_2w, marker='o', label='2 Workers')
plt.plot(data_sizes, training_times_3w, marker='o', label='3 Workers')




plt.title('Computation Time vs Data Size % by number of Worker Nodes (XGBoost Classifier)')
plt.xlabel('Data Size (% of 10M rows)')
plt.ylabel('Computation Time (minutes)')
plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# Plotting the result

data_sizes_millions = [20, 40, 60, 80, 100]

training_times =  [162.4202, 350.3280, 509.2504, 672.5889, 814.2559]
time_in_mins = [round((time / 60), 2) for time in training_times]

plt.figure(figsize=(10, 6))
plt.plot(data_sizes_millions, time_in_mins, marker='o', linestyle='-', color='green')

for x, y in zip(data_sizes_millions, time_in_mins):
    plt.text(x, y + 0.2, f"{y:.2f} min", ha='center', fontsize=9)

plt.title('Computation Time vs Data Size % (XGBoost Classifier)')
plt.xlabel('Data Size (% of 10M rows)')
plt.ylabel('Training Time (Minutes)')
plt.grid(True)
plt.xticks(data_sizes_millions)
plt.tight_layout()
plt.show()



# COMMAND ----------


# Data
data_sizes = ['20', '40', '60', '80', '100']
accuracy = [0.6754, 0.6784, 0.6988, 0.7169, 0.7143]

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(data_sizes, accuracy, color='skyblue')

# Add accuracy values on top of each bar
for bar, acc in zip(bars, accuracy):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

# Titles and labels
plt.title('Accuracy vs Data Size (XGBoost Classifier)', fontsize=13)
plt.xlabel('Data Size (%)')
plt.ylabel('Accuracy')
plt.ylim(0.65, 0.73)
plt.tight_layout()
plt.show()

# COMMAND ----------


# Data
data_sizes = ['2M', '4M', '6M', '8M', '10M']

precision = [0.6849, 0.6864, 0.7054, 0.7226, 0.7201]
recall = [0.9494, 0.9557, 0.9687, 0.9766, 0.9757]

# X-axis configuration
x = np.arange(len(data_sizes))
width = 0.25

# Plotting
plt.figure(figsize=(10, 6))
bar2 = plt.bar(x, precision, width, label='Precision', color='lightgreen')
bar3 = plt.bar(x + width, recall, width, label='Recall', color='salmon')

# Adding value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.002, f'{height:.4f}', ha='center', va='bottom', fontsize=9)


add_labels(bar2)
add_labels(bar3)

# Labels and legend
plt.xlabel('Data Size')
plt.ylabel('Score')
plt.title('Precision, and Recall vs Data Size (XGBoost Classifier, 2 Workers)')
plt.xticks(x, data_sizes)
plt.ylim(0.65, 1.00)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

