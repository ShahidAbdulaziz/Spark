# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 617 –HW 08Instructions
# MAGIC **SHahid Abdulaziz**

# COMMAND ----------

import math
import pandas as pd
import numpy as np
import pyspark as py
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
import time

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md ## Problem 1: Creating Streaming DataFrame

# COMMAND ----------

step  = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .option('inferSchema',True)
    .csv('/FileStore/tables/paysim/step_001.csv')
)

StepSchema = step.schema

step.show(5)

# COMMAND ----------

paysim_stream = (
     spark.readStream
    .option('header', True)
    .option('maxFilesPerTrigger', 1)
    .schema(StepSchema)
    .csv('/FileStore/tables/paysim/')
)


print(paysim_stream.isStreaming)

# COMMAND ----------

# MAGIC %md ## Problem 2: Apply Transformations

# COMMAND ----------

type_summary = (
                paysim_stream
                            .groupBy('type')
                            .agg(
                                expr('count(type) AS n'),
                                expr('round(avg(amount),2) AS avg_amount'),
                                expr('min(amount) AS min_amount'),
                                expr('max(amount) AS max_amount')
                                 )
                            .sort('n', ascending = False)

                )



destinations = (
                paysim_stream
                            .filter(expr('type == "TRANSFER"'))
                            .groupBy('nameDest')
                            .agg(
                                expr('count(type) AS n'),
                                expr('sum(amount) AS total_amount'),
                                expr('round(avg(amount),2) AS avg_amount')
                                 )
                            .sort('n', ascending = False)

                )




# COMMAND ----------

# MAGIC %md ## Problem 3: Define Output Sinks

# COMMAND ----------

type_summary = (
    type_summary
    .writeStream
    .format('memory')
    .queryName('type_summary')
    .outputMode('complete')
)
 




destinations = (
    destinations
    .writeStream
    .format('memory')
    .queryName('destinations')
    .outputMode('complete')
)
 


# COMMAND ----------

# MAGIC %md ## Problem 4: Start andMonitor the Streams

# COMMAND ----------

type_query = type_summary.start()
dest_query = destinations.start()

# COMMAND ----------

print(spark.sql('SELECT * from type_summary').count())
spark.sql('SELECT * from type_summary').show( truncate=False)

# COMMAND ----------

print(spark.sql('SELECT * from destinations').count())
spark.sql('SELECT * from destinations').show(16, truncate=False)

# COMMAND ----------

type_query.stop()
dest_query.stop()

# COMMAND ----------


