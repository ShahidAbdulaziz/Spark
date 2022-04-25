# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 617–Project 03Instructions
# MAGIC ## Forest Cover Prediction
# MAGIC ** Shahid Abdulaziz**

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Part A: Set up Environment
# MAGIC 
# MAGIC In this section I will be importing the packages needed for this assignment.

# COMMAND ----------

import math
import pandas as pd
import numpy as np
import pyspark as py
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
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
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part B: Load and Explore the Data
# MAGIC In this section we will be loading in the dataset to explore the data

# COMMAND ----------

Wilderness_schema  = (
    'Elevation INTEGER, Aspect INTEGER, Slope INTEGER, Horizontal_Distance_To_Hydrology INTEGER, '
    'Vertical_Distance_To_Hydrology INTEGER, Horizontal_Distance_To_Roadways INTEGER, Hillshade_9am INTEGER, Hillshade_Noon INTEGER, ' 
    'Hillshade_3pm INTEGER, Horizontal_Distance_To_Fire_Points INTEGER, Wilderness_Area STRING, Soil_Type INTEGER,Cover_Type INTEGER ')
 

fc  = (
    spark.read
    .option('delimiter', '\t')
    .option('header', True)
    .schema(Wilderness_schema)
    .csv('/FileStore/tables/forest_cover.txt')
)


    
fc.printSchema()

# COMMAND ----------

columnnames = fc.columns
fc[columnnames[0:6]].show(3)

# COMMAND ----------

N = fc.count()
print(N)

# COMMAND ----------

fc.groupBy('Cover_Type').agg(expr('count(Cover_Type) AS Count')).select('Cover_Type', F.round((col('Count')/N).alias('prop'),4)).show()


# COMMAND ----------

# MAGIC %md ## Part C: Preprocessingand Splitting the Data
# MAGIC In this section, we will be preparing the data for our models.

# COMMAND ----------

#https://maryville.instructure.com/courses/56731/files/10976108?wrap=1&fd_cookie_set=1
num_features = ['Elevation',
                'Aspect',
                'Slope',
                'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways',
                'Hillshade_9am',
                'Hillshade_Noon',
                'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']

cat_features = ['Wilderness_Area',
                'Soil_Type']
ix_features = [c + '_ix' for c in cat_features]
vec_features = [c + '_vec' for c in cat_features]
encoder = OneHotEncoder(inputCols=ix_features, outputCols=vec_features, dropLast= False)
feature_indexer = StringIndexer(inputCols=cat_features, outputCols=ix_features)
assembler_lr = VectorAssembler(inputCols=num_features + vec_features, outputCol='features_lr') 
assembler_dt = VectorAssembler(inputCols=num_features + ix_features, outputCol='features_dt') 

# COMMAND ----------

trainPipe =  Pipeline(stages=[feature_indexer,  encoder, assembler_lr,assembler_dt]).fit(fc)
fc_proc = trainPipe.transform(fc)
fc_proc.persist()
fc_proc.select(['features_dt', 'Cover_Type']).show(5, truncate=False)

# COMMAND ----------

splits = fc_proc.randomSplit([0.8, 0.2], seed=1)
train = splits[0]
test = splits[1]
train.persist()


print("Training Observations:", train.count())
print("Testing Observations:", test.count())

# COMMAND ----------

# MAGIC %md ## Part D: Hyperparameterfor Logistic Regression

# COMMAND ----------

accuracy_eval = MulticlassClassificationEvaluator(
    predictionCol='prediction', labelCol='Cover_Type', metricName='accuracy')

logreg = LogisticRegression(featuresCol='features_lr', labelCol='Cover_Type')

param_grid = (ParamGridBuilder()
              .addGrid(logreg.regParam, [ 0.00001, 0.0001, 0.001, 0.01, 0.1])
              .addGrid(logreg.elasticNetParam, [0, 0.5,1])
             ).build()
cv = CrossValidator(estimator=logreg, estimatorParamMaps=param_grid, evaluator=accuracy_eval, 
                    numFolds=5, seed=1, parallelism=8)

cv_model = cv.fit(train)

# COMMAND ----------

lr_model = cv_model.bestModel
opt_regParam = lr_model.getRegParam()
opt_enetParam = lr_model.getElasticNetParam()

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

# MAGIC %md
# MAGIC ## Part E: Hyperparameter Tuningfor Decision Trees
# MAGIC In this section, we will be constructing the best deicison tree model

# COMMAND ----------

accuracy_eval = MulticlassClassificationEvaluator(
    predictionCol='prediction', labelCol='Cover_Type', metricName='accuracy')

dtree = DecisionTreeClassifier(featuresCol='features_dt', labelCol='Cover_Type', 
                               maxDepth=3,maxBins=38, minInstancesPerNode=8, seed=1)

param_grid = (ParamGridBuilder()
              .addGrid(dtree.maxDepth, [2, 4, 6, 8, 12, 14, 16,18,20,22,24])
              .addGrid(dtree.minInstancesPerNode, [1, 2, 4])
             ).build()


cv = CrossValidator(estimator=dtree, estimatorParamMaps=param_grid, 
                    evaluator=accuracy_eval, numFolds=5, seed=1)
cv_model = cv.fit(train)

# COMMAND ----------

dt_model = cv_model.bestModel
maxDepth = dt_model.getMaxDepth()
minInstancesPerNode = dt_model.getMinInstancesPerNode()
 
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

features = num_features+cat_features

pd.DataFrame({
    'feature':features,
    'importance': dt_model.featureImportances
})

# COMMAND ----------

# MAGIC %md ## Part F: Identifying and Evaluating the Final Model
# MAGIC In this section, we will determine which of the two models above are the best

# COMMAND ----------

# MAGIC %md I would select the decision tree model because its accuracy rate is the better of the two with .7775.

# COMMAND ----------

test_pred = dt_model.transform(test)
test_pred.select('probability', 'prediction','Cover_Type').show(truncate=False)

# COMMAND ----------

pred_and_labels = test_pred.rdd.map(lambda x:(x['prediction'],float(x['Cover_Type'])))

# COMMAND ----------

metrics = MulticlassMetrics(pred_and_labels)
print('Test Set Accuracy:', round(metrics.accuracy,4))

# COMMAND ----------

label = [1,2,3,4,5,6,7]
confusion = metrics.confusionMatrix().toArray().astype(int)
pd.DataFrame(
    data=confusion,
    columns=label,
    index=label
)

# COMMAND ----------

# MAGIC %md
# MAGIC Observations in the test set with Cover Type 1 were misclassified by the model as Cover Type 2 a total of 101 times. This was the most common type of misclassification in the test set. 

# COMMAND ----------


print('Label   Precision   Recall')
print('--------------------------')
for i, lab in enumerate(label):
    print(f'{lab:<8}{metrics.precision(i+1):<12.4f}{metrics.recall(i+1):.4f}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 1. Which cover type is most likely to be correctly classified by our final model?
# MAGIC 
# MAGIC   Label 7 is most liekly to be correct
# MAGIC   
# MAGIC 2. Which cover type is most likely to be misclassified by our final model?
# MAGIC 
# MAGIC   Label 2 is most likely to misclassified 
# MAGIC   
# MAGIC 3. Which cover type has the greatest difference between its precision and recall? Explain the meaning of both of these values with respect to this cover type. 
# MAGIC 
# MAGIC   Covery type 5 has the greatest difference between its precision and recall. Precision is the percentage out of all positive predicted values and recall is the percentage of positive values out of predicitve values.
# MAGIC   

# COMMAND ----------


