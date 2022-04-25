# Databricks notebook source
# MAGIC %md # DSCI 617 –Homework 06
# MAGIC ** Shahid Abdulaziz**

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

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md ## Problem 1: Load Stroke Data

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

stroke_df.show(10)

# COMMAND ----------

N = stroke_df.count()
print(N)

# COMMAND ----------

(stroke_df
           .select('*')
           .groupBy('stroke')
           .agg(
                F.round((F.count(col('stroke'))/N),4).alias('prop')
               )
           .show()


)

# COMMAND ----------

# MAGIC %md # Problem 2: Preprocessing

# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
num_features = ['age','avg_glucose_level','bmi']
cat_features = ['gender','hypertension','heart_disease','ever_married','work_type','residence_type','smoking_status']

# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
ix_features = [c + '_ix' for c in cat_features]
vec_features = [c + '_vec' for c in cat_features]



# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
label_indexer = StringIndexer(inputCol='stroke', outputCol='label').setHandleInvalid("keep") 
feature_indexer = StringIndexer(inputCols=cat_features, outputCols=ix_features)

# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
encoder = OneHotEncoder(inputCols=ix_features, outputCols=vec_features, dropLast= False)

# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
assembler = VectorAssembler(inputCols=num_features + vec_features, outputCol='features')


# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
trainPipe =  Pipeline(stages=[label_indexer,feature_indexer, encoder, assembler]).fit(stroke_df)
train = trainPipe.transform(stroke_df)

train.persist()
train.select(['features', 'stroke']).show(10, truncate=False)


# COMMAND ----------

# MAGIC %md ## Problem 3: Hyperparameter Tuningfor Logistic Regression

# COMMAND ----------

#Source:https://maryville.instructure.com/courses/56731/files/10976119?wrap=1&fd_cookie_set=1
accuracy_eval = MulticlassClassificationEvaluator(
    predictionCol='prediction', labelCol='label', metricName='accuracy')

logreg = LogisticRegression(featuresCol='features', labelCol='label')

param_grid = (ParamGridBuilder()
              .addGrid(logreg.regParam, [ 0.0001, 0.001, 0.01, 0.1, 1])
              .addGrid(logreg.elasticNetParam, [0, 0.5,1])
             ).build()
cv = CrossValidator(estimator=logreg, estimatorParamMaps=param_grid, evaluator=accuracy_eval, 
                    numFolds=5, seed=1, parallelism=8)

cv_model = cv.fit(train)

# COMMAND ----------

#Source: https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
model = cv_model.bestModel
opt_regParam = model.getRegParam()
opt_enetParam = model.getElasticNetParam()

print('Max CV Score:  ', round(max(cv_model.avgMetrics),4))
print('Optimal Lambda:', opt_regParam)
print('Optimal Alpha: ', opt_enetParam)

# COMMAND ----------

model_params = cv_model.getEstimatorParamMaps()

lr_cv_summary_list = []
for param_set, acc in zip(model_params, cv_model.avgMetrics):
    new_set = list(param_set.values()) + [acc]
    lr_cv_summary_list.append(new_set)
lr_cv_summary = pd.DataFrame( lr_cv_summary_list, columns=['reg_param', 'enet_param', 'acc'])
for en in  lr_cv_summary.enet_param.unique():
    sel =  lr_cv_summary.enet_param == en
    plt.plot( lr_cv_summary.reg_param[sel],  lr_cv_summary.acc[sel], label=en)
    plt.scatter( lr_cv_summary.reg_param[sel],  lr_cv_summary.acc[sel]) 
plt.legend()
plt.xscale('log')
plt.grid()
plt.xlabel('Regularization Parameter')
plt.ylabel('Cross-Validation Score')
plt.show()
    

# COMMAND ----------

# MAGIC %md ## Problem 4: Training Predictions

# COMMAND ----------

train_pred = model.transform(train)
train_pred.select('probability', 'prediction', 'label').show(10, truncate=False)

# COMMAND ----------

train_pred.select('probability', 'prediction', 'label').filter(expr('prediction <> stroke')).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC * The highest probability observed for an incorrect answer is 0.7327
# MAGIC * The lowest probability observed for an incorrect answer is 0.3182 

# COMMAND ----------

# MAGIC %md ## Problem 5: Classification Metrics

# COMMAND ----------

pred_and_labels = train_pred.rdd.map(lambda x:(x['prediction'],float(x['stroke'])))

# COMMAND ----------

#Source: https://maryville.instructure.com/courses/56731/files/10976095?wrap=1&fd_cookie_set=1
metrics = MulticlassMetrics(pred_and_labels)
print(metrics.accuracy)

# COMMAND ----------

#Source: https://maryville.instructure.com/courses/56731/files/10976095?wrap=1&fd_cookie_set=1
cm = metrics.confusionMatrix().toArray().astype(int)
labels = trainPipe.stages[0].labels

# COMMAND ----------

pd.DataFrame(
    data=cm, 
    columns=labels,
    index=labels
)
 

# COMMAND ----------

#source: https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
print('cut         Precision   Recall')
print('------------------------------')
for i, lab in enumerate(labels):
    print(f'{lab:<12}{metrics.precision(i):<12.4f}{metrics.recall(i):.4f}')

# COMMAND ----------

# MAGIC %md ## Problem 6: Applying the Model to New Data

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


