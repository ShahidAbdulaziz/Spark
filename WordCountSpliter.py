# Databricks notebook source
# MAGIC %md #DSCI 617 – Homework 02
# MAGIC **Shahid Abdulaziz**

# COMMAND ----------

import math
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
from string import punctuation
from operator import add

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md ##Problem 1: Word Count
# MAGIC 
# MAGIC In the next few problems, we will work with a text file that contains the complete works of William Shakespeare.
# MAGIC The data file using for this problem is located at: /FileStore/tables/shakespeare_complete.txt.

# COMMAND ----------

# MAGIC %md
# MAGIC Complete the following steps in a single code cell:
# MAGIC 1. Read the contents of the file shakespeare_complete.txt into an RDD named ws_lines.
# MAGIC 2. Create an RDD named ws_words by applying the transformations described below. This will require
# MAGIC several uses of map() and flatMap() and a single call to filter(). Try to chain together the
# MAGIC transformations together to complete all of these steps with a single statement (that will likely span
# MAGIC multiple lines).
# MAGIC 
# MAGIC • Tokenize the strings in ws_lines by splitting them on the 8 characters in the following list:
# MAGIC [' ', '-', '_', '.', ',', ':', '|', '\t']
# MAGIC 
# MAGIC The resulting RDD should consist of strings rather than lists of strings. This will require
# MAGIC multiple separate uses of flatMap() and split().
# MAGIC 
# MAGIC • Use the Python string method strip() with the punctuation string to remove common
# MAGIC punctuation symbols from the start and end of the tokens. Then use strip() again with the
# MAGIC string '0123456789' to remove numbers from the start and end of the tokens.
# MAGIC (Code cell continued on next page.)
# MAGIC (Code cell continued from previous page.)
# MAGIC 
# MAGIC • Use the Python string method replace() to replaces instances of the single
# MAGIC quote/apostrophe "'" with the empty string ''.
# MAGIC 
# MAGIC • Convert all strings to lower case using the lower() string method.
# MAGIC 
# MAGIC • The steps above will create some empty strings of the form '' within the RDD. Filter out
# MAGIC these empty strings.
# MAGIC 
# MAGIC 3. Create a second RDD named dist_words that contains only one copy of each word found in
# MAGIC ws_words.
# MAGIC 4. Print the number of words in ws_words and the number of distinct words using the format shown
# MAGIC below. Add spacing so that the numbers are left-aligned.

# COMMAND ----------

ws_lines = sc.textFile('/FileStore/tables/shakespeare_complete.txt')

ws_words = (
     ws_lines
    .flatMap(lambda x : x.split(' '))      
    .flatMap(lambda x : x.split('-'))
    .flatMap(lambda x : x.split('_'))
    .flatMap(lambda x : x.split('.')) 
    .flatMap(lambda x : x.split(',')) 
    .flatMap(lambda x : x.split('|')) 
    .flatMap(lambda x : x.split('\t'))
    .flatMap(lambda x : x.split(':')) 
    .map(lambda x : x.strip(punctuation))
    .map(lambda x : x.strip('0123456789'))  
    .map(lambda x : x.replace("'", ''))   
    .map(lambda x : x.lower())  
    .filter(lambda x : x != '')            
)
dist_words = ws_words.distinct()





print("Total Number of Words: ", str(ws_words.count()))
print("Total Number of Words: ", str(dist_words.count()))





# COMMAND ----------

print(ws_words.sample(withReplacement=False,  fraction=0.0001).collect())


# COMMAND ----------

# MAGIC %md ##Problem 2: Longest Words
# MAGIC We will now find the longest words used by Shakespeare. We will start by looking for the single longest word. 

# COMMAND ----------

# MAGIC %md Complete the following steps in a single code cell:
# MAGIC 
# MAGIC 1. Write a Python function with two parameters, both of which are intended to be strings. The function should return the longer of the two strings. If the strings are the same length, then the function should return the word that appears later when ordered lexicographically (alphabetically). 
# MAGIC 
# MAGIC 2. Use the function you wrote along with reduce()to find the longest word in the RDD dist_words. 
# MAGIC 
# MAGIC Print the result. 

# COMMAND ----------

def String_Count(firstString, secondString):
       
    if len(firstString) > len(secondString):
        return firstString
        
    elif len(secondString) > len(firstString):
        return secondString
    
    else:
        return min(firstString,secondString)
    
        
  

        
    

# COMMAND ----------

dist_words.reduce(String_Count)

# COMMAND ----------

# MAGIC %md We will now find the 20 longest words used by Shakespeare. Use sortBy()with the Python len()function to sort the elements of dist_wordsaccording to their length, with longer words appearing first.
# MAGIC 
# MAGIC Print the first 20 elements of this RDD. 

# COMMAND ----------

dist_words.sortBy(lambda x :len(x),  ascending= False).take(20)

# COMMAND ----------

# MAGIC %md ## Problem 3: Word Frequency 
# MAGIC We will now create a frequency distribution for the words appearing in our document in order to determine which wordswere used most frequently by Shakespeare.  

# COMMAND ----------

# MAGIC %md Complete the following steps in a single code cell:
# MAGIC 1. Create an RDD named pairs. This RDD should consist of tuples of the form (x, 1), where xis a word in ws_words. The RDD pairsshould contain one element for each element of ws_words. 
# MAGIC 2. Use reduceByKey()to group the pairs together according to their first elements (the words), summing together theintegers stored in the second element (the 1s). This will produce an RDD with one pair for each distinct word. The first element will be the word and the second element will be a count for that word. Sort this RDD by the second tuple element (the count), in descending order. Name the resulting RDD word_counts. 
# MAGIC 3. Store the first 20 elements of word_countsin a list. Then use that list to create a Pandas DataFrame with two columns named "Word"and "Count". 
# MAGIC 4. Display this DataFrame (without using the print()function).

# COMMAND ----------

pairs = ws_words.map(lambda x: (x,1)).reduceByKey(add)

word_counts = pairs.sortBy(lambda x: x[1],ascending = False)

topWords = word_counts.take(20)    

pd.DataFrame(topWords, columns =['Word','Count'])


# COMMAND ----------

# MAGIC %md ##Problem 4: Removing Stop Words
# MAGIC You will notice that, unsurprisingly, the words most frequently used by Shakespeare are very common words such as "the" and "and". We will removethese common wordsand then recreate our frequency distribution. Words that are filtered out prior to performing a text-based analysis are referred to as stop words. There is no commonly accepted definition of what is and what is not a stop word, and the definitionused could vary by task. A document containing a list of stop words to use in this assignment has been providedat the following path: /FileStore/tables/stopwords.txt. The document contains one word per line. 

# COMMAND ----------

# MAGIC %md Complete the following steps in a single code cell:
# MAGIC 1. Read the contents of the file stopwords.txtinto an RDD named sw_rdd.  
# MAGIC 2. Print the number of elements in this RDD. 
# MAGIC 3. To get a sense as to the contents of the RDD, display a sample of elements contained within it. Perform the sampling without replacement and set fraction=0.05.
# MAGIC 4. Store the full contents of sw_rddin a list named sw. 

# COMMAND ----------

sw_rdd = sc.textFile('/FileStore/tables/stopwords.txt')
sw = sw_rdd.collect()

print("The number of elements are: ", sw_rdd.count())
print("The samples are: ", sw_rdd.sample(False, .05, seed=None).collect())

# COMMAND ----------

# MAGIC %md Create an RDD named ws_words_f by removing from ws_words the elements that are also contained in the swlist. Then create an RDD named dist_words_fconsisting of only the distinct elements from ws_words_f. Print the number of distinct non-stop words using the following format:
# MAGIC 
# MAGIC Number of Distinct Non-Stop Words: xxxx
# MAGIC 
# MAGIC We will now recreate our frequency distribution using only non-stop words.
# MAGIC 
# MAGIC Repeat the steps from Problem 3using ws_words_frather than ws_words.

# COMMAND ----------


ws_words_f  = (
     ws_words
    .filter(lambda x : x not in sw)
)


dist_words_f = ws_words_f.distinct()
print("Number of Distinct Non-Steop Words: ", dist_words_f.count())




# COMMAND ----------

ws_words_f = (
    ws_words
    .filter(lambda x : x not in sw)
    .map(lambda x : (x, 1))
    .reduceByKey(lambda x, y : x + y)
    .sortBy(lambda x : x[1], ascending=False)
)
 
for row in ws_words_f.take(20):
    print(f'{row[0]:<12}{row[1]:>4}') 

# COMMAND ----------

# MAGIC %md ###Problem 5: Diamonds Dataset 
# MAGIC We will now use the Diamonds Dataset to get anidea ofhow you might use RDD to work with structured data.This problem will provide useful practice for working with RDDs, but it should be mentioned that the DataFrame class (which we will discuss laterin the course) is a much better tool the RDD class for working with structureddata. The file containing the data for this problem is located at the path: /FileStore/tables/diamonds.txtThis dataset contains information for nearly 54,000 diamonds sold in the United States. For each diamond, we have values for 10 variables. A description of each of the variables is provided below, in the order in which they appear in the file. The variables cut, color, and clarityare ordinal, or ranked categorical variables. The levels for these variables are provided below in order from worst to best. 
# MAGIC 
# MAGIC * caratWeight of the diamond.
# MAGIC * cutQuality of the cut. Levels: Fair, Good, Very Good, Premium, Ideal
# MAGIC *  colorDiamond color. Levels: J, I, H, G, F, E, D
# MAGIC * clarityA measure of diamond clarity. Levels: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
# MAGIC * depthTotal depth percentage 
# MAGIC * tableWidth of top of diamond relative to widest point 
# MAGIC * pricePrice in US dollars
# MAGIC * xLength in mm
# MAGIC * yWidth in mm 
# MAGIC * zDepth in mm You can find more information about the Diamonds datasethere: Diamonds Data. 

# COMMAND ----------

# MAGIC %md Read the contents of the tab-delimitated file diamonds.txtinto an RDD named diamonds_raw. Print the number of elements in this RDD. 

# COMMAND ----------

diamonds_raw = sc.textFile('/FileStore/tables/diamonds.txt')
print(diamonds_raw.count())

# COMMAND ----------

# MAGIC %md In a new cell, use a loop and the take()action to display the first five elements of diamonds_raw. 

# COMMAND ----------

for row in diamonds_raw.take(5):
    print(row)

# COMMAND ----------

# MAGIC %md Create a function named process_rowwith a single parameter row, which is intended to accept strings representing (non-header) lines from the diamonds data file. The function should tokenize the line by splitting it on tab character'\t', and then return a list of the individual tokens coerced into the correct data types. The data types for the tokens are, in order
# MAGIC 
# MAGIC :float, str, str, str, float, float, int, float, float, float
# MAGIC 
# MAGIC Use filter()to remove the header row from diamonds_raw.Then use map()to apply process_row()to each element of the filtered RDD. Store the results in a variablenamed diamonds. 
# MAGIC 
# MAGIC Use a loop and the take()action to print the first 5 elements of this RDD. 

# COMMAND ----------

def process_line(row):
    obj = row.split('\t')
    if 'carat' in row:
        return obj  
    
    return(float(obj[0]),
           str(obj[1]),
           str(obj[2]),
           str(obj[3]),
           float(obj[4]),
           float(obj[5]),
           int(obj[6]),
           float(obj[7]),
           float(obj[8]),
           float(obj[9]),       
          )


diamonds = (
    diamonds_raw
    .filter(lambda x: 'carat' not in x)
    .map(process_line)


)
 
for row in diamonds.take(5):
    print(row)

    
    


# COMMAND ----------

# MAGIC %md ### Problem 6: Problem 6: Grouped Means
# MAGIC A diamond’s cut is a categorical feature describing how well-proportioned the dimensions of the diamond are. This feature has five possible levels. These levels are, in increasing order of quality, Fair, Good, Very Good, Premium, and Ideal. We will now use pair RDD tools to calculate the count, average price, and average carat size for diamonds with each of the five levels of cut. Note thatfor any tuple within the diamondsRDD:
# MAGIC * The caratsize for the associated diamond is stored at index 0 of the tuple.
# MAGIC * The cutlevel for the associated diamond is stored at index 1 of the tuple.
# MAGIC * The pricefor the associated diamond is stored at index 6 of the tuple.

# COMMAND ----------

# MAGIC %md Create a list named cut_summary by performing the transformations and action described below. Try to perform all of the stepswitha single (multi-line) statement by chaining together the methods. 
# MAGIC * Transform each observation into a tuple of the form (cut, (carat, price, 1)). Note that the first element of this tuple indicates the cut level (which we will be grouping by), while the second element of the tuple is another tuple containing other information in which we are interested. 
# MAGIC * Use reduceByKey()to perform an elementwise sum of the tuples (carat, price, 1)for each separate value of the key, which is represented by the cutvalue. This will produce an RDD with 5 elements of the form (cut, (sum_of_carat, sum_of_price, count)).
# MAGIC * Use map()to transform the tuples in the previous RDD into ones with the following form:(cut, count, mean_carat_size, mean_price). Round the two means to 2 decimal places. 
# MAGIC * Call the collect()method to create the desired 5 element list. 
# MAGIC 2. To better display the results, use cut_summaryto create a Pandas DataFrame named cut_df.Set the following names for the columns of the DataFrame: Cut, Count, Mean_Carat, Mean_Price.
# MAGIC 3. Display cut_df(without using the print()function).

# COMMAND ----------

cut_summary = (
diamonds
    .map(lambda x: (x[1] , [x[0], x[6]] ))
    .mapValues(lambda x : x + [1])
    .reduceByKey(lambda x, y : [a+b for a,b in zip(x,y)])
    .mapValues(lambda x : [x[-1]] + [round(a/x[-1],2) for a in x[:-1]])
    .map(lambda x : [x[0]] + x[1])
    .collect()


)

pd.DataFrame(cut_summary, columns = ("Cut","Count","Mean_Carat","Mean_Price"))

