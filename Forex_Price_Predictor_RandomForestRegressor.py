# Databricks notebook source
from pyspark.sql.functions import col, lag, avg, expr, to_timestamp, date_format
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

import time
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("Pinal_BigData_Project") \
    .getOrCreate()

filename = "dbfs:/FileStore/shared_uploads/psoni4@stevens.edu/chunk_10M.csv"
df = spark.read.format("csv").option("header", "true").load(filename)
df.show()

#Converting from string datatype to double datatype
df = df.withColumn("AskVolume", col("AskVolume").cast(DoubleType()))
df = df.withColumn("BidVolume", col("BidVolume").cast(DoubleType()))

# COMMAND ----------

# Step 2: Feature engineering

df = df.withColumn("MidPrice", (col("AskPrice") + col("BidPrice")) / 2)
df = df.withColumn("mid_lag1", lag("MidPrice", 1).over(window))
df = df.withColumn("mid_return", (col("MidPrice") - col("mid_lag1")) / col("mid_lag1"))
df = df.withColumn("mid_ma5", avg("MidPrice").over(window.rowsBetween(-4, 0)))


# Window spec for lag/rolling
window = Window.orderBy("UTC")

# Convert the UTC string to a timestamp
df = df.withColumn("UTC_timestamp", to_timestamp("UTC", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))

# Extract the date and hour components
df = df.withColumn("date", date_format("UTC_timestamp", "yyyy-MM-dd"))
df = df.withColumn("hour", date_format("UTC_timestamp", "HH"))

# Define WindowSpec for partitioning by both date and hour
window = Window.partitionBy("date", "hour").orderBy("UTC_timestamp")

df = df.withColumn("mid_lag1", lag("MidPrice", 1).over(window))
df = df.withColumn("mid_return", (col("MidPrice") - col("mid_lag1")) / col("mid_lag1"))
df = df.withColumn("mid_ma5", avg("MidPrice").over(window.rowsBetween(-4, 0)))

# Drop nulls from lag/moving avg
df = df.dropna()

# COMMAND ----------

# Step 3: Assemble features into one vector column named 'features' 
assembler = VectorAssembler(
    inputCols=["mid_lag1", "mid_return", "mid_ma5", "AskVolume", "BidVolume"],
    outputCol="features"
)

# COMMAND ----------

# Step 4: Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# COMMAND ----------

# Step 5: Model
rf = RandomForestRegressor(
    featuresCol="scaledFeatures",
    labelCol="MidPrice",
    predictionCol="prediction"
)

# COMMAND ----------

# Step 6: Pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# COMMAND ----------

# Step 7: Param grid for tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Step 8: Evaluator
evaluator = RegressionEvaluator(labelCol="MidPrice", predictionCol="prediction", metricName="rmse")

# Step 9: CrossValidator
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=2,             # k-fold cross-validation
    parallelism=2           # parallel grid search
)

# COMMAND ----------

# Step 10: Train/Test Chronological Split

# train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

split_ts = df.approxQuantile("UTC_timestamp", [0.8], 0.0)[0]
train_data = df.filter(df.UTC_timestamp <= split_ts)
test_data  = df.filter(df.UTC_timestamp > split_ts)

# COMMAND ----------

# Step 11: Train

start = time.time()
model = cv.fit(train_data)
end = time.time()

print(f"Elapsed time to train the random forest regressor for price prediction: {end-start:.4f} seconds")

# COMMAND ----------

# Step 12: Predict
predictions = model.transform(test_data)

# COMMAND ----------

predictions.select("UTC", "MidPrice", "prediction").show()

# COMMAND ----------

# RMSE (Root Mean Squared Error)
rmse_evaluator = RegressionEvaluator(
    labelCol="MidPrice", predictionCol="prediction", metricName="rmse")
rmse = rmse_evaluator.evaluate(predictions)

# MAE (Mean Absolute Error)
mae_evaluator = RegressionEvaluator(
    labelCol="MidPrice", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)

# R² (R-squared)
r2_evaluator = RegressionEvaluator(
    labelCol="MidPrice", predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# COMMAND ----------

# Plotting the result
data_sizes_millions = [20, 40, 60, 80, 100]

training_times =  [702.9470, 891.3144, 1102.3906, 1322.1354, 1494.9538]
time_in_mins = [round((time / 60), 2) for time in training_times]

plt.figure(figsize=(10, 6))
plt.plot(data_sizes_millions, time_in_mins, marker='o', linestyle='-', color='green')

for x, y in zip(data_sizes_millions, time_in_mins):
    plt.text(x, y + 0.2, f"{y:.2f} min", ha='center', fontsize=9)

plt.title('Computation Time vs Data Size % (Random Forest Regressor)')
plt.xlabel('Data Size (% of 10M rows)')
plt.ylabel('Training Time (Minutes)')
plt.grid(True)
plt.xticks(data_sizes_millions)
plt.tight_layout()
plt.show()



# COMMAND ----------

# Data sizes as percentage of 10M
data_percents = [20, 40, 60, 80, 100]

# Training times in seconds
worker_1_times = [1133.4202, 1520.6222, 1932.3472, 2331.8, 2731.3]
worker_2_times = [702.9470, 891.3144, 1102.3906, 1322.1354, 1494.9538]
worker_3_times = [611.2256, 633.2261, 671.2635, 735.2651, 852.4641]

worker_1_times = [round((time / 60), 2) for time in worker_1_times]
worker_2_times = [round((time / 60), 2) for time in worker_2_times]
worker_3_times = [round((time / 60), 2) for time in worker_3_times]

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(data_percents, worker_1_times, marker='o', label='1 Worker')
plt.plot(data_percents, worker_2_times, marker='o', label='2 Workers')
plt.plot(data_percents, worker_3_times, marker='o', label='3 Workers')

for x, y in zip(data_percents, worker_1_times):
    plt.text(x, y + 0.2, f"{y:.2f} min", ha='center', fontsize=9)

for x, y in zip(data_percents, worker_2_times):
    plt.text(x, y + 0.2, f"{y:.2f} min", ha='center', fontsize=9)


for x, y in zip(data_percents, worker_3_times):
    plt.text(x, y + 0.2, f"{y:.2f} min", ha='center', fontsize=9)

# # Add labels to each point
# for x, y in zip(data_percents, worker_1_times):
#     plt.text(x, y + 20, f"{y:.0f}s", ha='center', fontsize=9)
# for x, y in zip(data_percents, worker_2_times):
#     plt.text(x, y + 20, f"{y:.0f}s", ha='center', fontsize=9)
# for x, y in zip(data_percents, worker_3_times):
#     plt.text(x, y + 20, f"{y:.0f}s", ha='center', fontsize=9)

# Labels and legend
plt.title('Computation Time vs Data Size % by number of Worker Nodes (Random Forest Regressor)')
plt.xlabel('Data Size (% of 10M rows)')
plt.ylabel('Training Time (minutes)')
plt.xticks(data_percents)  # Only show the defined percentages
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Data sizes (in millions)
data_sizes = [20, 40, 60, 80, 100]

# RMSE values for each worker configuration
rmse_1_worker = [
    0.0004201788646268096,
    0.0004810324895809866,
    0.000467891842035658,
    0.0004553,
    0.00125763
]

rmse_2_worker = [
    0.0004162843726768011,
    0.0004918149646999021,
    0.00046897575381882344,
    0.00048540225846188315,
    0.0011747947265630515
]

rmse_3_worker = [
    0.0004197842399055883,
    0.0004819464276473407,
    0.00046895154071575863,
    0.0005142521778949109,
    0.0015858882654769737
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, rmse_1_worker, marker='o', label='1 Worker')
plt.plot(data_sizes, rmse_2_worker, marker='o', label='2 Workers')
plt.plot(data_sizes, rmse_3_worker, marker='o', label='3 Workers')

# Chart styling
plt.title('RMSE vs Data Size (%) for Different Worker Counts')
plt.xlabel('Data Size (% of 10M rows)')
plt.ylabel('RMSE')
plt.xticks(data_sizes)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Display plot
plt.show()

# COMMAND ----------

