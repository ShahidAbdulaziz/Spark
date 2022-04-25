# Databricks notebook source
# MAGIC %md # DSCI 617 –Homework 07
# MAGIC **Shahid Abdulaziz**

# COMMAND ----------

import math
import pandas as pd
import numpy as np
import pyspark as py
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
from string import punctuation
from operator import add
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, LongType
import matplotlib.pyplot  as plt
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md ## Problem 1:Decision Tree Classification

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC <pre>
# MAGIC +--+----+----+----+----+-----------+|
# MAGIC x0 |  x1|  x2|  x3|  x4| prediction |  Leaf Node
# MAGIC +--+----+----+----+----+------------+  ---------|
# MAGIC 3.7| 5.6| 3.6| 2.0| 1.0|  0.0       |    3      |
# MAGIC 8.2| 4.2| 2.1| 2.0| 0.0|  0.0       |    7      | 
# MAGIC 5.4| 3.9| 4.9| 1.0| 1.0|  2.0       |    5      | 
# MAGIC 2.8| 6.1| 8.1| 0.0| 0.0|  2.0       |    2      +
# MAGIC ----+----+----+----+----+-----------+</pre>

# COMMAND ----------

# MAGIC %md ##Problem 2: Random Forest Classification

# COMMAND ----------

print("Tree Model 1 Prediction:  0.0")
print("Tree Model 2 Prediction:  0.0")
print("Tree Model 3 Prediction:  1.0")
print("Random Forest Prediction: 0.0")

# COMMAND ----------

# MAGIC %md ## Problem 3:Load and Process Stroke Data

# COMMAND ----------

stroke_schema  = (
    'gender STRING, age DOUBLE, hypertension INTEGER, heart_disease INTEGER, '
    'ever_married STRING, work_type STRING, residence_type STRING, avg_glucose_level DOUBLE, ' 
    ' bmi DOUBLE, smoking_status STRING, stroke INTEGER ')
 

stroke_df  = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(stroke_schema )
    .csv('/FileStore/tables/stroke_data.csv')
)


    
stroke_df .printSchema()

# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
num_features = ['age','avg_glucose_level','bmi']
cat_features = ['gender','hypertension','heart_disease','ever_married','work_type','residence_type','smoking_status']
ix_features = [c + '_ix' for c in cat_features]
feature_indexer = StringIndexer(inputCols=cat_features, outputCols=ix_features).setHandleInvalid("keep") 
assembler = VectorAssembler(inputCols=num_features + ix_features, outputCol='features')

# COMMAND ----------

trainPipe =  Pipeline(stages=[feature_indexer,  assembler]).fit(stroke_df)
train = trainPipe.transform(stroke_df)
train.persist()
train.select(['features', 'stroke']).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md ## Problem 4: Hyperparameter Tuning for Decision Trees

# COMMAND ----------

accuracy_eval = MulticlassClassificationEvaluator(
    predictionCol='prediction', labelCol='stroke', metricName='accuracy')

dtree = DecisionTreeClassifier(featuresCol='features', labelCol='stroke', 
                               maxDepth=3, minInstancesPerNode=8, seed=1)

param_grid = (ParamGridBuilder()
              .addGrid(dtree.maxDepth, [2, 4, 6, 8, 12, 14, 16])
              .addGrid(dtree.minInstancesPerNode, [4, 8, 16, 32])
             ).build()


cv = CrossValidator(estimator=dtree, estimatorParamMaps=param_grid, 
                    evaluator=accuracy_eval, numFolds=5, seed=1)
cv_model = cv.fit(train)

# COMMAND ----------

model = cv_model.bestModel
maxDepth = model.getMaxDepth()
minInstancesPerNode = model.getMinInstancesPerNode()
 
print('Max CV Score:   ', round(max(cv_model.avgMetrics),4))
print('Optimal Depth:  ', maxDepth)
print('Optimal MinInst:', minInstancesPerNode)

# COMMAND ----------

model_params = cv_model.getEstimatorParamMaps()
dt_cv_summary_list = []
for param_set, acc in zip(model_params, cv_model.avgMetrics):
    new_set = list(param_set.values()) + [acc]
    dt_cv_summary_list.append(new_set)
cv_summary = pd.DataFrame(dt_cv_summary_list, 
                          columns=['maxDepth', 'minInst', 'acc'])
    
for mi in cv_summary.minInst.unique():
    sel = cv_summary.minInst == mi
    plt.plot(cv_summary.maxDepth[sel], cv_summary.acc[sel], label=mi)
    plt.scatter(cv_summary.maxDepth[sel], cv_summary.acc[sel]) 
plt.legend()
plt.grid()
plt.xticks(range(2,18,2))
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Score')
plt.show()

# COMMAND ----------

# MAGIC %md ## Problem 5: Structure of Final Model

# COMMAND ----------

print(model.toDebugString)

# COMMAND ----------

features = num_features+cat_features
print(features)

# COMMAND ----------

print("First Feature Used in Tree: Age")
print("Features Unused in Tree:6: ever_married,7:work_type,8:residence_type")

# COMMAND ----------

pd.DataFrame({
    'feature':features,
    'importance':model.featureImportances
})

# COMMAND ----------

# MAGIC %md ## Problem 7

# COMMAND ----------

newData = [
    ['Female',  42.0  ,1  ,0  , 'No'  ,'Private'      ,'Urban'   ,182.1  ,26.8  ,'smokes'],
    ['Female',  64.0  ,0  ,1  , 'Yes' ,'Self-employed','Urban'   ,171.5  ,32.5  ,'formerly smoked'],
    ['Female',  37.0  ,0  ,0  , 'Yes' ,'Private'      ,'Urban'   ,79.2   ,18.4  ,'Unkown'],
    ['Female',  72.0  ,0  ,1  , 'No'  ,'Private'      ,'Govt_job',125.7  ,19.4  ,'never smoked']          
]
newData_schema  = (
    'gender STRING, age DOUBLE, hypertension INTEGER, heart_disease INTEGER, '
    'ever_married STRING, work_type STRING, residence_type STRING, avg_glucose_level DOUBLE, ' 
    'bmi DOUBLE, smoking_status STRING')


newData = spark.createDataFrame(data= newData, schema =newData_schema   )
newData.show()

# COMMAND ----------

newDataTrans = trainPipe.transform(newData)
newPred = model.transform(newDataTrans)
newPred.select('probability', 'prediction').show(truncate=False)

# COMMAND ----------


