# Databricks notebook source
dbutils.widgets.text("src_bucket", "dbfs:/databricks-datasets/airlines", "Source S3 Bucket")
airlines_bucket = dbutils.widgets.get("src_bucket")

# COMMAND ----------

# read from csv dataset
df = spark.read.option("header", "true").option("inferSchema", "true").csv(airlines_bucket+ "/part-00000")
df_schema = df.schema

# COMMAND ----------

# save the schema to s3
bucket_name = 'dbc-mwc'
prefix = 'expedia/staging/schemas/'
schema_file = prefix + 'airlines_schema.json'

def write_df_schema_s3(bucket, fname, schema):
  import boto3, json
  s3 = boto3.resource('s3')

  # Creating an empty file called "_DONE" and putting it in the S3 bucket
  return s3.Object(bucket, fname).put(Body=json.dumps(json.loads(schema.json())))
  
write_df_schema_s3(bucket_name, schema_file, df_schema)

# COMMAND ----------

def load_df_schema(bucket, fname):
  import boto3, json
  from pyspark.sql.types import StructType 

  # reload the schema from s3
  s3 = boto3.client('s3')

  obj = s3.get_object(Bucket=bucket, Key=fname)
  json_schema = obj['Body'].read().decode('utf-8')
  return StructType.fromJson(json.loads(json_schema))

new_schema = load_df_schema(bucket_name, schema_file)

# COMMAND ----------

bucket_files = list(map(lambda y: y.path, filter(lambda x: "part-" in x.name, dbutils.fs.ls(airlines_bucket))))[1:100]

# COMMAND ----------

df_full = spark.read.schema(new_schema).csv(bucket_files)

# COMMAND ----------

print("Number of records: {0}".format(df_full.count()))

# COMMAND ----------

spark.sql("""CREATE DATABASE IF NOT EXISTS mwc_prod""")

# COMMAND ----------

# write the full raw airlines dataset 
df_full.write.mode("overwrite").format("delta").saveAsTable("mwc_prod.airlines_raw")

# COMMAND ----------

df = spark.read.json("s3a://dbc-mwc/airports/")
df.write.mode("overwrite").format("delta").saveAsTable("mwc_prod.airport_codes")

# COMMAND ----------

spark.sql("""OPTIMIZE mwc_prod.airlines_raw""")
spark.sql("""OPTIMIZE mwc_prod.airport_codes""")