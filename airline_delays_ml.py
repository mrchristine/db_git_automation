# Databricks notebook source
# featurize the airlines dataset to do the following
# parse departure hour and cast as INT
# map IsDelayed to binary value

flightsDF = spark.sql("""
select 
  flightnum, 
  UniqueCarrier as uniquecarrier_cd, 
  Origin as origin_cd, 
  Dest as dest_cd,
  cast(substr(DepTime, 1, 2) as INT) as dephour, 
  CASE WHEN 
    IsDepDelayed = 'YES' 
      THEN 1 
    ELSE 0 
  END as delayed, 
  DayOfWeek as dayofweek 
from mwc_prod.airlines_raw """).na.fill(0.0).na.fill("N/A")
flightsDF.show(20)

# COMMAND ----------

# build features
features = list(filter(lambda x: x not in ["flightnum", "delayed"], flightsDF.columns))
print(features)

categoricals = list(filter(lambda x: "_cd" in x, features))
numerics = list(filter(lambda x: "_cd" not in x, features))

print(categoricals)
print(numerics)

# COMMAND ----------

from pyspark.ml.feature import *
string_indexers = [StringIndexer(inputCol=x, outputCol=x+"_IDX", handleInvalid="error", stringOrderType="frequencyDesc") for x in categoricals]
cat_cols = list(map(lambda x: x + "_IDX", categoricals))

# COMMAND ----------

va_numerics = VectorAssembler(inputCols=numerics, outputCol="numericFeatures")
va_categoricals = VectorAssembler(inputCols=cat_cols, outputCol="categoricalFeatures")

# COMMAND ----------

scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledNumericFeatures")

# COMMAND ----------

allFeatures = VectorAssembler(inputCols=["scaledNumericFeatures", "categoricalFeatures"], outputCol="features")
stages = string_indexers

stages.extend((va_numerics, scaler, va_categoricals, allFeatures))

stages

# COMMAND ----------

from pyspark.ml import Pipeline, PipelineModel
pipeline = Pipeline(stages=stages)

# COMMAND ----------

feature_model = pipeline.fit(flightsDF)

# COMMAND ----------

featurized_dataset = feature_model.transform(flightsDF)

# COMMAND ----------

# spark.sql("CREATE DATABASE IF NOT EXISTS mwc_ml")
# featurized_dataset.write.mode("overwrite").format("delta").saveAsTable("mwc_ml.flights_featurized")

# COMMAND ----------

# spark.sql("OPTIMIZE mwc_ml.flights_featurized")

# COMMAND ----------

featurized_ds = spark.table("mwc_ml.flights_featurized")
(trainingData, testData) = featurized_ds.randomSplit([0.7, 0.3], seed = 100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

lr = LogisticRegression(featuresCol="features", labelCol="delayed", regParam=0.01, weightCol="weight")

# COMMAND ----------

