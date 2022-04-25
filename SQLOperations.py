# Databricks notebook source
# MAGIC %md # DSCI 617 –Homework 04
# MAGIC **Shahid Abdulaziz**

# COMMAND ----------

# MAGIC %md ## Load Diamond Data

# COMMAND ----------

import math
import pandas as pd
import numpy as np
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

# COMMAND ----------

diamonds_schema  = (
    'carat DOUBLE, cut STRING, color STRING, clarity STRING, '
    'depth DOUBLE, table DOUBLE, price INTEGER, x DOUBLE, ' 
    ' y DOUBLE, z DOUBLE')
 

diamonds = (
    spark.read
    .option('delimiter', '\t')
    .option('header', True)
    .schema(diamonds_schema)
    .csv('/FileStore/tables/diamonds.txt')
)


    
diamonds.printSchema()

# COMMAND ----------

# MAGIC %md ## Problem 1: Grouping By Cut

# COMMAND ----------

CutDictionary = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 5, 'Ideal': 6}

def rank_cut(df):
    return CutDictionary.get(df)


# COMMAND ----------

rank_cut_udf = F.udf(rank_cut, IntegerType())

# COMMAND ----------

(diamonds
 .select(
     '*')
     
 .groupBy('cut')
 .agg(expr('count(cut) as n_diamonds'),
      expr( 'round(avg(price)) as avg_price'),
      expr( 'round(avg(carat),2) as avg_carat'),
      expr( 'round(avg(depth),2) as avg_depth'),
      expr( 'round(avg(table),2) as avg_table')    
     )
 .sort(rank_cut_udf(col('cut')).alias('value'))

 .show()
                
)
    


# COMMAND ----------

# MAGIC %md ## Problem 2: Filtering based on Carat Size

# COMMAND ----------


idx = 0
x = -1

for idx in range(0,6):
    intervalOne = x +1
    intervalTwo = intervalOne + 1
    
    print("The number of diamonds with carat size in range", (intervalOne,intervalTwo ),"is", diamonds.filter(col('carat') >= intervalOne).filter(col('carat') < intervalTwo).count())
    
    x += 1
    idx += 1
    





# COMMAND ----------

# MAGIC %md ## Problem 3: Binning by Carat Size

# COMMAND ----------

def carat_bin(df):
    x = -1
    idx = 0
    for idx in range(0,6):
        intervalOne = x +1
        intervalTwo = intervalOne + 1
        if df >= intervalOne  and df < intervalTwo:
            return( '['+str(intervalOne)+','+str(intervalTwo)+')')
        x += 1
        idx += 1
        
    

        
  
    
    
    
    
    



# COMMAND ----------

carat_bin_udf = F.udf(carat_bin, StringType())

# COMMAND ----------

(diamonds
 .select(
     '*',
    carat_bin_udf(col('carat')).alias('carat_bin'))
 .groupBy('carat_bin')
 .agg(expr('count(carat_bin) as n_diamonds'),
      expr( 'round(avg(price)) as avg_price')
     )
 .sort('carat_bin')

 .show()
                

) 

# COMMAND ----------

# MAGIC %md ## Load IMDB Data

# COMMAND ----------

movies = (
    spark.read
    .option('delimiter', '\t')
    .option('header', True)

    .csv('/FileStore/tables/imdb/movies.txt')
)

names = (
    spark.read
    .option('delimiter', '\t')
    .option('header', True)

    .csv('/FileStore/tables/imdb/names.txt')
)

title_principals = (
    spark.read
    .option('delimiter', '\t')
    .option('header', True)

    .csv('/FileStore/tables/imdb/title_principals.txt')
)

ratings = (
    spark.read
    .option('delimiter', '\t')
    .option('header', True)

    .csv('/FileStore/tables/imdb/ratings.txt')
)



# COMMAND ----------

movies.printSchema()
names.printSchema()
title_principals.printSchema()
ratings.printSchema()

# COMMAND ----------

print(movies.count())
print(names.count())
print(title_principals.count())
print(ratings.count())

# COMMAND ----------

# MAGIC %md ## Problem 4: Number of Appearances by Actor

# COMMAND ----------

(title_principals
     .select('*')
    .filter(expr('category == "actor" OR category == "actress" '))
    .groupBy('imdb_name_id')
    .agg(expr('count(imdb_name_id) AS appearences'))
    .join(other = names, on = 'imdb_name_id', how = 'inner')
    .select('name','appearences')
    .sort('appearences', ascending=False)
    .show(16)


)

# COMMAND ----------

# MAGIC %md ##Problem 5: Average Rating by Director

# COMMAND ----------

(title_principals
     .select('*')
    .filter(expr('category == "director" '))
    .join(other = ratings, on = 'imdb_title_id', how = 'inner')
    .groupBy('imdb_name_id')
    .agg(expr('count(imdb_title_id) as num_films'),
         expr('sum(total_votes) as total_votes'),
         expr('round(avg(rating),2) as avg_ratings'),
        )
    .filter(expr('total_votes >= 1000000 AND num_films >=5'))
    .join(other = names, on = 'imdb_name_id', how = 'inner')
    .select('name',
            'num_films',
            'total_votes',
            'avg_ratings')
    .sort('avg_ratings', ascending=False)
    .show(16,truncate=False)


)

# COMMAND ----------

ratings.show()

# COMMAND ----------

title_principals.show()

# COMMAND ----------

# MAGIC %md ## Problem 6: Actors Appearing in Horror Films

# COMMAND ----------

horror_films = movies.filter(expr('genre LIKE "%Horror%"'))
horror_films.count()

# COMMAND ----------

(title_principals
     .select('*')
    .filter(expr('category == "actor" OR category == "actress" '))
    .join(other = horror_films, on = 'imdb_title_id', how = 'inner')
    .groupBy('imdb_name_id')
    .agg(expr('count(imdb_title_id) as num_films'))
    .join(other = names, on = 'imdb_name_id', how = 'inner')
    .select('name',
            'num_films')
    .sort('num_films', ascending=False)
    .show(16)


)

# COMMAND ----------


