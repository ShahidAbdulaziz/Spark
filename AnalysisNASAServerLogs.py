# Databricks notebook source
# MAGIC %md #DSCI 617 – Project 01
# MAGIC ## Analysis of NASA Server Logs
# MAGIC **Shahid Abdulaziz**

# COMMAND ----------

import math
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
from string import punctuation
from operator import add
import pyspark as py
import matplotlib.pyplot  as plt
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc

# COMMAND ----------

# MAGIC %md ## Part A: Set up Environment
# MAGIC In this part of the project, we will set up our environment. We will begin with some import statements. 

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md ##Part B: Load and Process Data:
# MAGIC 
# MAGIC In this project, you will be working with a file containing one month of server log data collected fromNASA.gov in August 1995. This file is located at the path /FileStore/tables/NASA_server_logs_Aug_1995.txt.

# COMMAND ----------

nasa_raw = sc.textFile('/FileStore/tables/NASA_server_logs_Aug_1995.txt')

print("Total Number of Elements ", str(nasa_raw.count()))



for row in nasa_raw.take(10):
    print(row)


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 1. Create a function named process_row(). This function should accept a single parameter named row, which is intended to represent individual elements of the nasa_rawRDD. The function should perform the follow processing tasks on the string contained in row, in the order described.
# MAGIC   1. Use the string replace()method to replacedouble quotes with empty strings. 
# MAGIC   2. Tokenize the string on space characters using the split()method. 
# MAGIC   3. If the last token (indicating bytes) is equal to a hyphen, replace it with 0.
# MAGIC   4. Coerce the bytestoken to an integer. (Note that the status code could be interpreted as an integer, but we will leave it as a stringto more easily reflect that this is categorical information). e.Return the resulting list of tokens. 
# MAGIC 
# MAGIC 2. Apply the process_row()function to the elements of nasa_rawto create a new RDD named nasa. The new RDD should contain the same number of elements as nasa_raw, but these elements should be lists instead of strings. 
# MAGIC 3. Persist the nasaRDD to memory. 
# MAGIC 4. Print the first 10 elements of the nasaRDD, with each element appearing on a different line of output.

# COMMAND ----------

def process_row(row): 
    nasaSplit = row.replace('"', '').split(' ')
    if nasaSplit[-1] == '-':
        nasaSplit[-1] = 0
    return [str(nasaSplit[0]), str(nasaSplit[1]), str(nasaSplit[2]),
            str(nasaSplit[3]), str(nasaSplit[4]), int(nasaSplit[5])]


#(int(nasaSplit[-1]))

    
nasa = nasa_raw.map(process_row)   
nasa.persist(py.StorageLevel.MEMORY_ONLY) 

for row in nasa.take(10):
    print(row)


# COMMAND ----------

# MAGIC %md ## Part C: Most Requested Resources
# MAGIC In this part of the project, we will determine which resources were requested the most frequently. 

# COMMAND ----------

# MAGIC %md 
# MAGIC Create a pair RDD named count_by_resourceby performing the steps described below. Try to perform all of the steps in a single (multi-line) statement by chaining together RDD transformations. 
# MAGIC 
# MAGIC 1. Create a pair RDD with one element for each element of the nasaRDD. The elements of the new pair RDD should have the following form: (resource_location, 1)
# MAGIC 2. Use a pair RDD method to create an RDD with one element for each request type. The elements of this new RDD should have the following form: (resource_location, count)
# MAGIC 3. Sort the pair RDD from the previous step according to its valueelement(count). This sorted RDD should be stored in count_by_resource.
# MAGIC 
# MAGIC Print the first 10 elementsof the count_by_resourceRDD, with each element appearing on a different line of output.

# COMMAND ----------

count_by_resource = (
    nasa
    .map(lambda x: x[3])
    .map(lambda x: (x, 1))
    .reduceByKey(lambda x, y : x + y)
    .sortBy(lambda x : x[1], ascending=False)
    
    
    
    
)
for row in count_by_resource.take(10):
    print(row)

# COMMAND ----------

# MAGIC %md ## Part D: Most Common Request Origins
# MAGIC For each request, we are provided with the IP address or DNS hostname for the server from which the request originated. In this part of the project, we will determine which servers are the origins for the greatest number of requests. 

# COMMAND ----------

count_by_resource = (
    nasa
    .map(lambda x: x[0])
    .map(lambda x: (x, 1))
    .reduceByKey(lambda x, y : x + y)
    .sortBy(lambda x : x[1], ascending=False)
    
    
    
    
)
for row in count_by_resource.take(10):
    print(row)

# COMMAND ----------

# MAGIC %md ## Part E: Request Types
# MAGIC In this part of the project, we will analyze records based on their request type. We will start by confirming that there are three different request types. 

# COMMAND ----------

# MAGIC %md 
# MAGIC Create a list that containing one occurrence of each value that appears as a request type within the elements of the nasaRDD. You will need to use the map() method to select out the request type, and then use the distinct() method to determine the unique request types. Store the list in a variable named req_types and then print the contents of this list. 

# COMMAND ----------

req_types = nasa.map(lambda x: x[2]).distinct().collect()
print(req_types)

# COMMAND ----------

# MAGIC %md 
# MAGIC Loop over the elements of the req_typeslist. For each request type, determine the number of records with that particular request type. Print the results in the format shown below. The only RDD action that should be used in this cell is the count()action. You can use whatever RDD transformations are needed for the task.
# MAGIC 
# MAGIC There were xxxx GET requests.
# MAGIC 
# MAGIC There were xxxx HEAD requests.
# MAGIC 
# MAGIC There were xxxx POST requests.

# COMMAND ----------

idx = 0

for idx in range(0,len(req_types)): 
    print("There were ",nasa.filter(lambda x: x[2] == req_types[idx]).count(),req_types[idx], " request." )
    idx += 1



# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Create a pair RDD named avg_bytesby performing the steps described below. Try to perform all of the steps in a single (multi-line) statement by chaining together RDD transformations.
# MAGIC 1. Create a pair RDD with one element for each element of the nasaRDD. The elements of the new pair RDD should have the following form: (request_type, (bytes, 1))
# MAGIC 2. Use a pair RDD method to create an RDD with one element for each request type. The elements of this new RDD should have the following form: (request_type, (total_bytes, count))
# MAGIC 3. Use a pair RDD methodto create an RDD with one element for each request type. The elements of this new RDD should have the following form: (request_type, avg_bytes_per_request). The second element of this RDD should be rounded to the nearest integer. This final RDD is should be stored in avg_bytes.
# MAGIC 
# MAGIC Print each element of the avg_bytesRDD, with each element appearing on a different line of output.

# COMMAND ----------

avg_bytes = (
nasa
    .map(lambda x: (x[2] , [x[5]] ))
    .mapValues(lambda x : x + [1])
    .reduceByKey(lambda x, y : [a+b for a,b in zip(x,y)])
    .mapValues(lambda x : [x[-1]] + [round(a/x[-1],2) for a in x[:-1]])
    .map(lambda x : [x[0]] + x[1])
    .map(lambda x: (x[0],x[2]))
    





)
for row in avg_bytes.take(10):
    print(row)

# COMMAND ----------

# MAGIC %md ## Part F: Status Codes
# MAGIC In this part of the project, we will analyze the status codes returned by the server. 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Create a list that containing one occurrence of each value that appears as a status code within the elements of the nasaRDD, sorted in ascending order. Use RDD methodsto determine the contentsof this list and also to perform the sorting. Store the resulting list in a variable named status_codesand then print this list. 

# COMMAND ----------

status_codes = nasa.map(lambda x: x[4]).distinct().collect()
print(status_codes)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Loop over the elements of the req_typeslist created in Part E. For each request type, perform the tasks described below. Attempt to perform steps 1 –5 with a single statement by chaining together RDD methods. 
# MAGIC 1. Apply a filter to the nasaRDD keeping only elements of the current request type.
# MAGIC 2. Extract the status codes for these records.
# MAGIC 3. Determine the collectionof distinct status codes appearing for this request type.
# MAGIC 4. Sort the distinct codes in increasing order.
# MAGIC 5. Collect the results into a list. 
# MAGIC 6. Print a message stating which status codes appear for this request.
# MAGIC 
# MAGIC Your final output should be formatted as follows:
# MAGIC 
# MAGIC Status codes for GET requests:  LIST_OF_STATUS_CODES
# MAGIC 
# MAGIC Status codes for HEAD requests: LIST_OF_STATUS_CODES
# MAGIC 
# MAGIC Status codes for POST requests: LIST_OF_STATUS_CODES

# COMMAND ----------

idx = 0
for idx in range(0,len(req_types)): 
    print("Status codes for ", req_types[idx]  ," requests: ",nasa.filter(lambda x: x[2] == req_types[idx]).map(lambda x: x[4]).distinct().collect())
    idx += 1

# COMMAND ----------

# MAGIC %md 
# MAGIC Perform the following steps in a single code cell.
# MAGIC 1. Create an empty list named code_counts.
# MAGIC 2. Loop over the elements of the status_codes list. For each status code, determine the number of requests resulting in that particular status code. Append the resulting count into the code_countslist.
# MAGIC 3. Use the contents of status_codesand code_countsto create a bar chart displaying the number of requests resulting in each status code. Construct your bar chart according to the following specifications
# MAGIC  1.  The figure size should be [10,4].
# MAGIC  2. Select a single named colorto use for the bars. Add a black border to each bar.
# MAGIC  3. The chart should be titled “Distribution of Status Codes”d.The x-axis should be labeled “Status Code” and the y-axis should be labeled “Count”. 
# MAGIC  4. The heights of the bars will be on dramatically different scales. For this reason, we will use a log scale for they-axis. This can be set using plt.yscale('log').
# MAGIC  5. Use plt.show()to display the figure. 

# COMMAND ----------

code_counts = []

idx = 0

for idx in range(0,len(status_codes)): 
    
    
    code_counts.append((nasa.filter(lambda x: x[4] == status_codes[idx]).count()))
    idx += 1


plt.figure(figsize=[10,4])
plt.bar(status_codes, code_counts, color='green')
plt.title('Distribution of Status Codes')
plt.xlabel('Status Code')
plt.ylabel('Count')
plt.yscale('log')
plt.show()

# COMMAND ----------

# MAGIC %md ## Part G: Request Volume by Day
# MAGIC In the final part of this project, we will determine the number of requests received by the server during each day in August 1995. 

# COMMAND ----------

# MAGIC %md
# MAGIC Create a list named counts_by_dayby performing the steps described below. Attempt to perform all of these tasks with a single (multi-line) statement by chaining together RDD methods. 
# MAGIC 1. Create a pair RDD with one element for each element of the nasaRDD. The elements of this RDD should have the form (day_of_month, 1). For example, if a request was made on the 13th of August, the element associated with this request would look like ('13', 1). Note that the day of the month is containedin the second and third characters of the date/time element for each request. Also note that string objects in Python can be sliced similar to lists. For example, if my_stringis a Python string, then my_string[1:3]would return the second and third characters of that string.
# MAGIC 2. Use a pair RDD method to create a new RDD with one record for each day of the month. The elements of this new RDD should have the form: (day_of_month, count_of_requests)
# MAGIC 3. Sort the previous pair RDD in increasing order by its key values(that is, by the day of the month).
# MAGIC 4. Collect the result into a list named counts_by_day.
# MAGIC 
# MAGIC Print the first five elements of the counts_by_daylist. These should be tuples containing two elements. The first should be a string representing a day of the month and the second should be the number of requests received by the server on that day. 

# COMMAND ----------

counts_by_day =  (
    nasa
    .map(lambda x: x[1][1:3])
    .map(lambda x: (x, 1))
    .reduceByKey(lambda x, y : x + y)
    .sortBy(lambda x : x[0], ascending=True)
            
)

for row in counts_by_day.take(20):
    print(row)


# COMMAND ----------

# MAGIC %md
# MAGIC Perform the following steps in a single code cell.
# MAGIC 1. Separate the elements of the tuples within counts_by_dayinto two lists named day_listand count_list. The first elements of the tuples should form day_listand the second elements should be used to form count_list. This can be accomplished using a single loop or two list comprehensions.
# MAGIC 2. Use the contents of day_listand count_listto create a bar chart displaying the number of requests received on each day. Construct your bar chart according to the following specifications:
# MAGIC   1. The figure size should be [10,4].
# MAGIC   2. Select a single named colorto use for the bars. Add a black border to each bar.
# MAGIC   3. The chart should be titled “Number of Requests by Day (Aug 1995)”
# MAGIC   4. The x-axis should be labeled “Day of Month” and the y-axis should be labeled “Count”. 
# MAGIC   5. Use plt.show()to display the figure. 

# COMMAND ----------

day_list = counts_by_day.map(lambda x: str(x[0])).collect()
count_list = counts_by_day.map(lambda x: int(x[1])).collect()



# COMMAND ----------

plt.figure(figsize=[10,4])
plt.bar(day_list, count_list, color='green')
plt.title('Number of Requests by Day (Aug 1995)')
plt.xlabel('Day of Month')
plt.ylabel('Count')
plt.yscale('log')
plt.show()

# COMMAND ----------


