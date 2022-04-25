# Databricks notebook source
# MAGIC %md 
# MAGIC #DSCI 617 –Project 02
# MAGIC ## Student Grade Database
# MAGIC ** Shahid Abdulaziz**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Set up Environment
# MAGIC In this part of the project, we will set up our environment. We will begin with some import statements. 

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

# MAGIC %md 
# MAGIC ## Part B: Load the Data
# MAGIC In this project, you will be working with a database of course records from a (fictional) university called SyntheticusUniversity, which was founded in Fall 2000. The university offers six undergraduate degree programs: Biology, Chemistry, Computer Science, Mathematics, and Physics. The data we will work with is assumed to have been collected immediately after the end of the Spring 2021 term. 

# COMMAND ----------

# MAGIC %md
# MAGIC Use the data files provided to create DataFrames with the following names:accepted,alumni, courses, expelled, faculty, grades, and unretained. Create a custom schema for each DataFrame. For compactness of code, I recommend using a DDL string to define the schemas. Columns named sid, fid, or creditsshould be integers and all other columns should be strings. 

# COMMAND ----------

accepted_schema  = (
    'acc_term_id STRING, sid INTEGER, first_name STRING, last_name STRING, '
    'major STRING')
alumni_schema  = (
    ' sid INTEGER')
 
courses_schema  = (
    'dept STRING, course STRING, prereq STRING, credits INTEGER')
expelled_schema  = (
    ' sid INTEGER')
faculty_schema  = (
    'fid INTEGER, first_name STRING, last_name STRING,dept STRING ')
grades_schema  = (
    'term_id STRING, course STRING, sid INTEGER, fid INTEGER, '
    'grade STRING')
unretained_schema  = (
    ' sid INTEGER')

 

accepted = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(accepted_schema)
    .csv('/FileStore/tables/univ/accepted.csv')
)

alumni = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(alumni_schema)
    .csv('/FileStore/tables/univ/alumni.csv')
)

courses = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(courses_schema)
    .csv('/FileStore/tables/univ/courses.csv')
)

expelled = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(expelled_schema)
    .csv('/FileStore/tables/univ/expelled.csv')
)

faculty = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(faculty_schema)
    .csv('/FileStore/tables/univ/faculty.csv')
)

grades = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(grades_schema)
    .csv('/FileStore/tables/univ/grades.csv')
)

unretained = (
    spark.read
    .option('delimiter', ',')
    .option('header', True)
    .schema(unretained_schema)
    .csv('/FileStore/tables/univ/unretained.csv')
)


    


# COMMAND ----------

# MAGIC %md 
# MAGIC Next, we will print the number of records in each DataFrame. 

# COMMAND ----------

print("The number of records in accepted is", str(accepted.count()) + '.')
print("The number of records in alumni is", str(alumni.count()) + '.')
print("The number of records in courses is", str(courses.count()) + '.')
print("The number of records in faculty is", str(faculty.count()) + '.')
print("The number of records in grades is", str(grades.count()) + '.')
print("The number of records in unretained is", str(unretained.count()) + '.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Student Count by Status
# MAGIC In this part, we will count the number of students in each of the following groups: students who have been accepted, students who actually enrolledin courses, current students, all former students, alumni, unretained students, and studentswho were expelled. 

# COMMAND ----------

# MAGIC %md We will now create a three new DataFrames to store student info for students in various categories. We will then generate the desired counts. 

# COMMAND ----------

enrolled = (
     accepted
         .select('*')
         .join(other = grades, on = 'sid', how = 'semi') 
 )

current = (
    enrolled
         .select('*')
         .join(other = alumni, on = 'sid', how = 'anti')
         .join(other = unretained, on = 'sid', how = 'anti')
         .join(other = expelled, on = 'sid', how = 'anti')

 )

former = (
    enrolled
         .select('*')
         .join(other = current, on = 'sid', how = 'anti')
 )

print("Number of accepted students:", str(accepted.count()) + '.')
print("Number of enrolled students:", str(enrolled.count()) + '.')
print("Number of current students:", str(current.count()) + '.')
print("Number of former students:", str(former.count()) + '.')
print("Number of unretained students:", str(unretained.count()) + '.')
print("Number of expelled students:", str(expelled .count()) + '.')
print("Number of alumni students:", str(alumni.count()) + '.')


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part D: Distribution of Students by Major
# MAGIC In this part, we will determine of the number of students currently in each major, as well as the proportion of the overall number of students in each major.

# COMMAND ----------

totalNumStudents = current.count()
numStudentByMajor = (
         current

                .groupBy('major')
                .agg(expr("count(sid) AS n_students"),
                 F.round(F.count(col('sid'))/totalNumStudents,4).alias('prop')                    
                    )
                .sort('prop', ascending = False)
)
numStudentByMajor.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part E: Course Enrollments by Department
# MAGIC In this part, we will determine of the number of students enrolled in courses offered by each department duringthe Spring 2021term. Recall that this term is encoded as '2021A'.

# COMMAND ----------

# MAGIC %md 
# MAGIC Determine the number of course enrollments during the Spring 2021 term. This is equal to the number of records in the grades DataFrame associated with that term. Store the count in a variable named sp21_enr.

# COMMAND ----------

 sp21_enr = grades.filter(expr('term_id = "2021A" ')).count()
    
(
grades
    .filter(expr('term_id = "2021A" '))
    .join(other = courses, on = 'course', how = 'inner')
    .select('*')
    .groupBy('dept')
    .agg(
            F.count('sid').alias('n_students'),
            F.round(F.count(col('sid'))/sp21_enr,4).alias('prop')  
        )
    .sort('prop', ascending = False)

           


).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part F: Graduation Rates by Major
# MAGIC 
# MAGIC In this part, we will determine the graduation rates for each major. We will perform this analysis in steps. First, we will create a DataFrame containing the number of former students in each major. Then we will create a DataFrame containing the number of alumni for each major. We will then combine these DataFrames to determine the graduation rate. 

# COMMAND ----------

# MAGIC %md 
# MAGIC Starting with the formerDataFrame, perform grouping and aggregation operations to determine the number of former students from each major. The resulting DataFrame should contain one record for each major, and have two columns named major, and n_former. The DataFrame should be sorted by major, in increasing order. 

# COMMAND ----------

former_by_major =        (
            former
                .groupBy('major')
                .agg(
                        F.count('sid').alias('n_former')
                    )
                .sort('n_former', ascending = True)

            )
former_by_major.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC We will now determine the number of alumni for each major. 

# COMMAND ----------

alumni_by_major =  (
                    former
                        .join(other = alumni, on = 'sid', how = 'semi')
                        .groupBy('major')
                        .agg(
                                F.count('sid').alias('n_alumni')
                            )
                        .sort('n_alumni', ascending = True)




                    )

alumni_by_major.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC We will now use the previous two DataFrames to determine the graduation rates. 

# COMMAND ----------

(
                    former_by_major
                        .join(other = alumni_by_major, on = 'major', how = 'inner')
                        .select("*"
                                ,expr('round(INT(n_alumni) / INT(n_former),4) AS grad_rate')

                                )                               
                                                                             
                        .sort('grad_rate', ascending = True)




).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Part G: Number of Terms Required for Graduation
# MAGIC In this part, we will find a frequency distribution for the number of terms that alumni required for graduation. 

# COMMAND ----------

(
grades
    .join(other = alumni, on = 'sid', how = 'semi')
    .groupBy('sid')
    .agg(
        expr(' count(distinct term_id) AS n_terms')
    )
    .groupBy('n_terms')
    .agg(
        expr('count(sid) AS n_alumni '))
    .sort('n_terms')


).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part H: Current Student GPA
# MAGIC In this section, we will calculate the GPA of each current student at SU and will analyze the results. 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Create a Python function that converts letter grades ('A', 'B','C', 'D', and 'F') to numerical grade (4, 3, 2, 1, and 0). Register this function as a Spark UDF. 

# COMMAND ----------

gradeValues = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}


def gradeConversion(df):
    return gradeValues.get(df)

gradeConversion_udf = F.udf(gradeConversion, IntegerType())

# COMMAND ----------

# MAGIC %md 
# MAGIC We will now calculate the GPA of each student currently enrolled at SU. This process is described below. The approached described is fairly straight-forward, but there are other (in some cases, more concise) ways of accomplishing the task described below. Feel free to explore different approach

# COMMAND ----------

current_gpa = (
grades
    .join(other = courses, on = 'course', how = 'left')
    .select('*',
            gradeConversion_udf(col('grade')).alias('num_grade'),
            (col('credits')* gradeConversion_udf(col('grade'))).alias('gp')
           )
    .groupBy('sid')
    .agg(F.sum('gp'),
        F.sum('credits'),
        F.round((F.sum('gp')/ F.sum('credits')),2).alias('gpa')
        )
    .join(other = current, on = 'sid', how = 'inner')
    .select(
            'sid',
            'first_name',
            'last_name',
            'major',
            'gpa')
    .sort('GPA', ascending = True)
)

current_gpa.show(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC Calculate and display a count of the number of current students with perfect 4.0 GPAs

# COMMAND ----------

current_gpa.filter(expr('gpa = 4.0')).count()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a histogram displaying the distribution of GPAs for current students. 

# COMMAND ----------

plt.hist(current_gpa.toPandas().gpa,color = 'green', bins= np.arange(0, 4+.25, .25 ),edgecolor='black' )
plt.xlabel('GPA')
plt.ylabel('Count')
plt.title('GPA Distribution for Current Students')
plt.legend()
plt.show

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part I: Grade Distribution by Instructor

# COMMAND ----------

# MAGIC %md 
# MAGIC In this part, we will determine the proportion of A, B, C, D, and F grades given out by each faculty member at SU. 

# COMMAND ----------

faculty_grade_dist = (grades
                      
                          .groupBy('fid')
                          .agg(
                                  F.count('fid').alias('N'),
                                  expr('SUM(CASE WHEN grade == "A" THEN 1 ELSE 0 END)AS countA'),
                                  expr('SUM(CASE WHEN grade == "B" THEN 1 ELSE 0 END)AS countB'),
                                  expr('SUM(CASE WHEN grade == "C" THEN 1 ELSE 0 END)AS countC'),
                                  expr('SUM(CASE WHEN grade == "D" THEN 1 ELSE 0 END)AS countD'),
                          
                          )
                          .join(other = faculty, on = 'fid', how = 'left')
                          .select('fid',
                                  'first_name',
                                  'last_name',
                                  'dept',
                                  'N',
                                  expr('round(countA/N,2) as PropA'),
                                  expr('round(countB/N,2) as PropB'), 
                                  expr('round(countC/N,2) as PropC'),
                                  expr('round(countD/N,2) as PropD'),
                      
                                 )
                      
                      
                     )
faculty_grade_dist.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC We will now identify the 10 faculty members who assign the fewest A grades. 

# COMMAND ----------

faculty_grade_dist.filter(expr('N >=100')).sort('propA', ascending = True).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC We will now identify the 10 faculty members who award A’s most frequently. 

# COMMAND ----------

faculty_grade_dist.filter(expr('N >=100')).sort('propA', ascending = False).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part J: First Term GPA
# MAGIC In this section, we calculate the first-term GPA for each student who has enrolledin classesat SU. 

# COMMAND ----------

first_term_gpa = (
grades
    .join(other = accepted, on = 'sid', how = 'left')
    .filter(expr('term_id = acc_term_id '))
    .join(other = courses, on = 'course', how = 'left')
     .select('*',
                
                    (col('credits')* gradeConversion_udf(col('grade'))).alias('gp')
                    )
     .groupBy('sid')
     .agg(
          F.round((F.sum('gp')/ F.sum('credits')),2).alias('first_term_gpa')
         )
      .sort('first_term_gpa', ascending = False)


)
first_term_gpa.show(10)    

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Part K: Graduation Rates and First Term GPA
# MAGIC In this section, we will calculate graduation rates for students whose first term GPA falls into each of four different grade ranges. 

# COMMAND ----------

# MAGIC %md
# MAGIC Create a Python function accepts a single parameter, gpa.

# COMMAND ----------

def gpa_bin(gpa):
    x = -1
    idx = 0
    for idx in range(0,4):
        intervalOne = x +1
        intervalTwo = intervalOne + 1
        if gpa == 4:
            return( '['+str(intervalOne)+','+str(intervalTwo)+')')
        elif gpa >= intervalOne  and gpa < intervalTwo:
            return( '['+str(intervalOne)+','+str(intervalTwo)+')')
        x += 1
        idx += 1
        
    
gpa_bin_udf = F.udf(gpa_bin, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC We will now calculate the number of alumni whose first-term GPA falls into each bin. 

# COMMAND ----------

alumni_ft_gpa = first_term_gpa.join(other = alumni, on = 'sid', how = 'semi').select('*', gpa_bin_udf(col('first_term_gpa')).alias('gpa_bin')).groupBy('gpa_bin').agg(F.count(col('gpa_bin')).alias("n_alumni")).sort('gpa_bin').filter(expr('gpa_bin is not NULL'))
alumni_ft_gpa.show()
alumni.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC Next, we will determine the number of former students whose first-term GPA falls into each bin. 

# COMMAND ----------

former_ft_gpa = (
                first_term_gpa
                                .join(other = former, on = 'sid', how = 'semi')
                                .select('*', 
                                        gpa_bin_udf(col('first_term_gpa')).alias('gpa_bin'))
                                .groupBy('gpa_bin').agg(F.count(col('gpa_bin')).alias("n_former"))
                                .sort('gpa_bin')
                                .filter(expr('gpa_bin is not NULL'))
)
former_ft_gpa.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We will now use the previous two DataFrames to determine the graduation rates for each of the GPA bins. 

# COMMAND ----------

(
alumni_ft_gpa
    .join(other = former_ft_gpa, on = 'gpa_bin', how = 'left')
    .select(
            '*',
            expr('round(n_alumni/n_former,2) as grad_rate')
            )
    .sort('gpa_bin', ascending = True)
    .show()

)

# COMMAND ----------


