import sys
import warnings
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.mllib.tree import GradientBoostedTreesModel
from pyspark.mllib.evaluation import RegressionMetrics

warnings.filterwarnings("ignore")

def display_banner():
    print("=" * 80)
    print("                        ENERGY CONSUMPTION PREDICTOR")
    print("=" * 80)
    print("A Spark-powered application for estimating energy usage on unseen datasets,")
    print("leveraging a pre-trained Gradient Boosted Trees regression model.\n")

def show_step(message):
    """Displays step-by-step progress messages."""
    print(f"[STEP] {message}...".ljust(80, '.'), "Completed")

def show_performance(rmse_val, r2_val):
    """Prints out performance metrics in a structured format."""
    print("\n" + "=" * 70)
    print("                   Prediction Performance Report")
    print("=" * 70)
    print(f"{'Root Mean Squared Error (RMSE)':<40} {rmse_val:>20.4f}")
    print(f"{'R2 (Coefficient of Determination)':<40} {r2_val:>20.4f}")
    print("=" * 70 + "\n")

def predict_energy_usage(file_path):
    display_banner()
    
    # Initialize SparkSession
    show_step("Setting up Spark environment")
    spark = SparkSession.builder \
        .appName('EnergyConsumptionPredictor') \
        .master('local') \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Load the dataset
        show_step("Importing test data")
        dataset = spark.read.csv(file_path, header=True, inferSchema=True)

        # Standardize column names
        show_step("Standardizing column names")
        for field in dataset.columns:
            dataset = dataset.withColumnRenamed(field, field.replace(' ', '_'))

        # Handle categorical features
        show_step("Transforming categorical features")
        categorical_features = [name for name, dtype in dataset.dtypes if dtype == 'string']
        for feature in categorical_features:
            distinct_vals = dataset.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
            distinct_vals = sorted(distinct_vals)
            encoding_map = {val: idx for idx, val in enumerate(distinct_vals)}

            def encode_category(val):
                return encoding_map.get(val, -1)

            encode_udf = udf(encode_category, IntegerType())
            dataset = dataset.withColumn(feature + '_encoded', encode_udf(dataset[feature]))

        # Remove original categorical columns
        show_step("Removing original categorical fields")
        processed_data = dataset.drop(*categorical_features)

        # Cast all remaining columns to float
        show_step("Casting columns to float")
        for feature in processed_data.columns:
            processed_data = processed_data.withColumn(feature, col(feature).cast('float'))

        # Prepare feature set and target
        show_step("Preparing features and target variable")
        target_feature = 'Energy_Consumption'
        predictors = [field for field in processed_data.columns if field != target_feature]

        # Convert DataFrame to RDD
        show_step("Converting DataFrame to RDD format")
        feature_rdd = processed_data.select(*predictors).rdd.map(lambda row: [float(x) for x in row])
        target_rdd = processed_data.select(target_feature).rdd.map(lambda row: float(row[0]))

        # Load the pre-trained model
        show_step("Retrieving pre-trained model")
        gbt_model = GradientBoostedTreesModel.load(spark.sparkContext, 'EnergyPredictorGBT')

        # Generate predictions
        show_step("Generating predictions")
        predicted_vals = gbt_model.predict(feature_rdd)

        # Combine predictions with true labels
        show_step("Aligning predictions with true values")
        prediction_label_pairs = predicted_vals.zip(target_rdd)

        # Evaluate the model
        show_step("Evaluating prediction performance")
        evaluator = RegressionMetrics(prediction_label_pairs)

        # Display performance metrics
        show_performance(evaluator.rootMeanSquaredError, evaluator.r2)

        print("=" * 80)
        print("                        ENERGY CONSUMPTION PREDICTOR COMPLETED")
        print("=" * 80)

    finally:
        # Stop SparkSession
        show_step("Terminating Spark environment")
        spark.stop()

if __name__ == "__main__":
    # If running from command-line, use this:
    if len(sys.argv) != 2:
        print("Usage: spark-submit predict.py <path_to_test_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    predict_energy_usage(file_path)
    

