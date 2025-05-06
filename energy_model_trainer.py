import warnings
import numpy as np
import pandas as pd
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.evaluation import RegressionMetrics

# Suppress warnings
warnings.filterwarnings("ignore")

# ========== APPLICATION HEADER ==========
print("\n" + "="*80)
print("                     ENERGY CONSUMPTION PREDICTION SYSTEM")
print("="*80)
print("This application loads energy usage data, preprocesses features,")
print("trains a Gradient Boosted Trees regression model, evaluates its performance,")
print("and saves the trained model for future deployment.\n")

# ========== STEP 1: INITIALIZE SPARK CONTEXT ==========
print("[Step 1] Initializing Spark Session...")

spark_configuration = SparkConf().setAppName('EnergyConsumptionPredictionModel').setMaster('local')
spark_context = SparkContext(conf=spark_configuration)
spark_session = SparkSession(spark_context)
spark_session.sparkContext.setLogLevel("ERROR")
print("Spark Session initialized successfully.\n")

# ========== STEP 2: LOAD DATASET ==========
print("[Step 2] Loading the Training Dataset...")

raw_dataset = spark_session.read.csv("TrainingDataset.csv", header=True, inferSchema=True)

# Replace spaces in column names for consistency
for column_name in raw_dataset.columns:
    raw_dataset = raw_dataset.withColumnRenamed(column_name, column_name.replace(' ', '_'))

print("Schema after standardizing column names:")
raw_dataset.printSchema()

row_count = raw_dataset.count()
column_count = len(raw_dataset.columns)

print(f"Dataset Loaded: {row_count} rows, {column_count} columns.\n")

# ========== STEP 3: ENCODE CATEGORICAL FEATURES ==========
print("[Step 3] Identifying and Encoding Categorical Variables...")

categorical_features = [column for column, dtype in raw_dataset.dtypes if dtype == 'string']
print(f"Categorical Features Detected: {categorical_features}\n")

for feature in categorical_features:
    unique_values = raw_dataset.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
    unique_values = sorted(unique_values)
    mapping_dictionary = {value: index for index, value in enumerate(unique_values)}
    print(f"Encoding '{feature}' with mapping: {mapping_dictionary}")

    def encode_category(value):
        return mapping_dictionary.get(value, -1)

    encoding_udf = udf(encode_category, IntegerType())
    encoded_column_name = feature + '_indexed'
    raw_dataset = raw_dataset.withColumn(encoded_column_name, encoding_udf(raw_dataset[feature]))

print("Categorical features successfully encoded.\n")

# ========== STEP 4: FINALIZE DATASET ==========
print("[Step 4] Finalizing Dataset for Model Training...")
print("Casting all columns to float...")
processed_dataset = raw_dataset.drop(*categorical_features)
for feature in processed_dataset.columns:
    processed_dataset = processed_dataset.withColumn(feature, col(feature).cast('float'))

print("Schema after feature transformation:")
processed_dataset.printSchema()
print()

# ========== STEP 5: PREPARE FEATURES AND LABELS ==========
print("[Step 5] Preparing Feature Vectors and Labels...")

target_variable = 'Energy_Consumption'
feature_variables = [feature for feature in processed_dataset.columns if feature != target_variable]

features_list = processed_dataset.select(*feature_variables).rdd.map(lambda row: [float(x) for x in row]).collect()
labels_list = processed_dataset.select(target_variable).rdd.map(lambda row: float(row[0])).collect()

labeled_data_points = [LabeledPoint(label, features) for label, features in zip(labels_list, features_list)]
full_dataset_rdd = spark_context.parallelize(labeled_data_points)

print(f"Prepared {len(labeled_data_points)} labeled data points.\n")

# ========== STEP 6: SPLIT DATA INTO TRAINING AND TESTING ==========
print("[Step 6] Splitting Data into Training and Testing Sets...")

training_data, testing_data = full_dataset_rdd.randomSplit([0.70, 0.30], seed=42)

print(f"Training Set: {training_data.count()} samples")
print(f"Testing Set : {testing_data.count()} samples\n")

# ========== STEP 7: TRAIN THE GRADIENT BOOSTED TREES MODEL ==========
print("[Step 7] Training the Regression Model...")

start_time = time.time()

gbt_model = GradientBoostedTrees.trainRegressor(
    training_data,
    categoricalFeaturesInfo={},
    numIterations=150,
    learningRate=0.2,
    maxDepth=2
)

training_duration = round(time.time() - start_time, 2)
print(f"Model training completed successfully in {training_duration} seconds.\n")

# ========== STEP 8: EVALUATE MODEL PERFORMANCE ==========
print("[Step 8] Evaluating Model on Testing Data...")

predictions = gbt_model.predict(testing_data.map(lambda x: x.features))
actual_vs_predicted = predictions.zip(testing_data.map(lambda x: x.label))

evaluation_metrics = RegressionMetrics(actual_vs_predicted)

print("-" * 80)
print(f"Root Mean Squared Error (RMSE): {evaluation_metrics.rootMeanSquaredError:.4f}")
print(f"R2 Score (Coefficient of Determination): {evaluation_metrics.r2:.4f}")
print("-" * 80 + "\n")

# ========== STEP 9: SAVE THE TRAINED MODEL ==========
print("[Step 9] Saving the Trained Regression Model...")

gbt_model.save(spark_context, 'EnergyPredictorGBT')
print("Model successfully saved to path: 'EnergyPredictorGBT'\n")

# ========== APPLICATION FOOTER ==========
print("\n" + "="*80)
print("              ENERGY CONSUMPTION PREDICTION SYSTEM COMPLETED")
print("="*80 + "\n")

# Gracefully shut down Spark
spark_context.stop()
print("Spark Context stopped successfully. Application terminated.\n")
