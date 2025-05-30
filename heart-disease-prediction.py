from pyspark.sql import SparkSession

#initialize spark session
spark = SparkSession.builder \
    .appName("Heart Disease Prediction") \
    .getOrCreate()

#load and inspect data
df = spark.read.csv("heart.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)

#data preprocessing
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# Drop rows with missing values
df = df.dropna()

# Identify categorical columns
categorical_cols = [field for (field, dtype) in df.dtypes if dtype == "string"]

# StringIndexer for categorical variables
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_cols]

# Assemble features into a single vector
feature_cols = [col for col in df.columns if col != "target" and col not in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols + [col+"_index" for col in categorical_cols], outputCol="features")

# Define the pipeline
pipeline = Pipeline(stages=indexers + [assembler])
df_prepared = pipeline.fit(df).transform(df)

#split data into training and test sets
train_data, test_data = df_prepared.randomSplit([0.8, 0.2], seed=42)

#train logistic regression model
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="target")
lr_model = lr.fit(train_data)

#evaluate the model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = lr_model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="target")
accuracy = evaluator.evaluate(predictions)
print(f"Test Area Under ROC: {accuracy}")

# Show predictions
predictions.select("features", "target", "prediction", "probability").show(10, truncate=False)

#stop the spark session
spark.stop()
