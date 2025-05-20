# Task 1: Big Data Analysis using PySpark

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BigDataAnalysis").getOrCreate()

# Load large dataset
df = spark.read.csv("nyc_taxi.csv", header=True, inferSchema=True)

# Show schema
df.printSchema()

# Basic analysis
print("Total Trips:", df.count())
df.select("fare_amount").groupBy().avg().show()

# Stop Spark session
spark.stop()
