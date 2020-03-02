# Databricks notebook source
import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
from matching_library import *
from modeling_library import *
from connection_utilities import SQLFunctions
import databricks.koalas as ks
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark.conf.set("spark.sql.crossJoin.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Matching 2 different datasets (SF & OS)

# COMMAND ----------

Connection = SQLFunctions(dbutils, spark)
sf_os_query = '''SELECT
          A.[Id],
		  A.[Name],
          A.[BillingStreet],
          A.[BillingCity],
          A.[BillingStateCode],
          A.[BillingPostalCode],
          A.[Phone],
          A.[BSD_Client_ID__c],
          C.[ClientKey],
          C.[ClientID],
		  C.[ClientName],
		  C.[AddressLine1],
          C.[AddressLine2],
          C.[CityName],
          C.[StateCode],
          C.[PostalCode],
          C.[BusinessPhoneNum]
	  FROM OneSource.Client C INNER JOIN Safesforce.Account A ON C.[ClientID] = A.[BSD_Client_ID__c]
      WHERE (C.[DataSourceCode] LIKE '%DimClient%') AND C.[CountryName] = 'United States' AND A.[BillingCountryCode] = 'US'
      '''

sf_os_data = Connection.read(query = sf_os_query)
sf_os = sf_os_data.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Preprocessing

# COMMAND ----------

# Turn ',' into None in AddressLine2 column
sf_os.loc[sf_os['AddressLine2'] == ',', 'AddressLine2'] = None

# COMMAND ----------

# Combine AddressLine1 and AddressLine2 into one column AddressLine
def combine_addresses(row):
  if row['AddressLine1']:
    if row['AddressLine2']:
      combined_address = row['AddressLine1'] + ' ' + row['AddressLine2']
    else:
      combined_address = row['AddressLine1']
  else:
     combined_address = None
  return combined_address

sf_os['AddressLine'] =  sf_os.apply(combine_addresses, axis = 1)

# COMMAND ----------

# get the basic cleaning dictionary and list from matching library
name_replace_dic = basic_name_replace_dic
name_remove_list = basic_name_remove_list
address_replace_dic = basic_address_replace_dic

# COMMAND ----------

# Name preprocessing
sf_os = text_preprocessing(sf_os, target_columns = ['Name', 'ClientName'], replace_dic = name_replace_dic, remove_list = name_remove_list)

#Address preprocessing
sf_os = text_preprocessing(sf_os, target_columns = ['BillingStreet', 'AddressLine'], replace_dic = address_replace_dic)
sf_os = text_preprocessing(sf_os, target_columns = ['BillingCity','BillingStateCode', 'CityName', 'StateCode'])

#Phone preprocessing
sf_os = number_preprocessing(sf_os, target_columns = ['Phone', 'BusinessPhoneNum'], lenght_limit = 10, cut_location = 'last', remove_consistent_number = True)

#Postal code preprocessing
sf_os = number_preprocessing(sf_os, target_columns = ['BillingPostalCode', 'PostalCode'], lenght_limit = 5, cut_location = 'first', remove_consistent_number = True)

# COMMAND ----------

# Drop the instances that have at least 1 missing value among selected columns
sf_os.dropna(subset = ['Id', 'BSD_Client_ID__c', 'Name_clean', 'BillingStreet_clean', 'BillingCity_clean', 'BillingStateCode_clean', 'Phone_clean', 'BillingPostalCode_clean', 
                      'ClientKey', 'ClientID', 'ClientName_clean', 'AddressLine_clean', 'CityName_clean', 'StateCode_clean', 'BusinessPhoneNum_clean', 'PostalCode_clean'], inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Generate matching table (cross join 2 tables)

# COMMAND ----------

# Splitting matched dataset into 2 datasets
sf = sf_os[['Id', 'BSD_Client_ID__c', 'Name_clean', 'BillingStreet_clean', 'BillingCity_clean', 'BillingStateCode_clean', 'Phone_clean', 'BillingPostalCode_clean']]
os = sf_os[['ClientKey', 'ClientID', 'ClientName_clean', 'AddressLine_clean', 'CityName_clean', 'StateCode_clean', 'BusinessPhoneNum_clean', 'PostalCode_clean']]

# COMMAND ----------

# Transform pandas dataframe to spark dataframe
sf_spark = spark.createDataFrame(sf)
os_spark = spark.createDataFrame(os)

# COMMAND ----------

# Cross join two tables to have all the possible combinations. Only implement within same states
sf_os_all_spark = sf_spark.join(os_spark, (sf_spark["BillingStateCode_clean"] == os_spark["StateCode_clean"]))

# COMMAND ----------

# Create a new spark dataframe that has both original information and the score columns
schema = StructType([StructField('Id', StringType(), True),
                     StructField('BSD_Client_ID__c', StringType(), True),
                     StructField('Name_clean', StringType(), True),
                     StructField('BillingStreet_clean', StringType(), True),
                     StructField('BillingCity_clean', StringType(), True),
                     StructField('BillingStateCode_clean', StringType(), True),
                     StructField('Phone_clean', StringType(), True),
                     StructField('BillingPostalCode_clean', StringType(), True),
                     StructField('ClientKey', IntegerType(), True),
                     StructField('ClientID', StringType(), True),
                     StructField('ClientName_clean', StringType(), True),
                     StructField('AddressLine_clean', StringType(), True),
                     StructField('CityName_clean', StringType(), True),
                     StructField('StateCode_clean', StringType(), True),
                     StructField('BusinessPhoneNum_clean', StringType(), True),
                     StructField('PostalCode_clean', StringType(), True),
                     StructField('name_score', DoubleType(), True),
                     StructField('street_score', DoubleType(), True),
                     StructField('city_score', DoubleType(), True),
                     StructField('phone_score', DoubleType(), True),
                     StructField('postalcode_score', DoubleType(),True)])  

# COMMAND ----------

# Using Pandas UDF here. Pandas UDF can distribute a spark dataframe and process them with pandas-based function
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)

# scores_assignment is the same function in matching library. But the two parameters fuzzy_columns and validation_columns are removed to the function inside, because the function used in Pandas UDF are only allowed to have input data as the parameter

def scores_assignment_in_spark(df):
  fuzzy_columns = {'name':['Name_clean','ClientName_clean', fuzz.token_sort_ratio], 'street':['BillingStreet_clean','AddressLine_clean', fuzz.token_sort_ratio]}
  validation_columns = {'city':['BillingCity_clean','CityName_clean'], 'phone':['Phone_clean','BusinessPhoneNum_clean'], 'postalcode':['BillingPostalCode_clean','PostalCode_clean']}
  
  if fuzzy_columns:
      for key in fuzzy_columns.keys():
          column_name = '{}_score'.format(key)
          df[column_name] = df.apply(lambda x: fuzzy_columns[key][2](x[fuzzy_columns[key][0]], x[fuzzy_columns[key][1]]), axis=1)

  if validation_columns:
      for key in validation_columns.keys():
          column_name = '{}_score'.format(key)
          df[column_name] = (df[validation_columns[key][0]] == df[validation_columns[key][1]]).astype(int)

  return df

matching_table_spark = sf_os_all_spark.groupby('Id').apply(scores_assignment_in_spark)

# COMMAND ----------

# Normalize the fuzzy matching score to (0,1)
matching_table_spark = matching_table_spark.withColumn("name_score", col("name_score")/100)
matching_table_spark = matching_table_spark.withColumn("street_score", col("street_score")/100)
# Generate a new column that is the summation of all the score
matching_table_spark = matching_table_spark.withColumn("scores_sum", col('name_score') + col('street_score') + col('city_score') + col('phone_score') + col('postalcode_score'))
# Cenerate label column, 1 means matching pair and 0 means not matching pair
matching_table_spark = matching_table_spark.withColumn("label", (col('BSD_Client_ID__c') == col('ClientID')).cast(IntegerType()))

# COMMAND ----------

# Sample the dataset into smaller size
sampled_matching_table_spark = matching_table_spark.sample(False, 0.001, seed=1234)

# COMMAND ----------

# Transform spark dataframe with selected columns to pandas dataframe
sampled_matching_table = sampled_matching_table_spark[['BSD_Client_ID__c', 'name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score', 'scores_sum', 'label']].toPandas()

# COMMAND ----------

X = sampled_matching_table[['BSD_Client_ID__c', 'name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score', 'scores_sum']]
y = sampled_matching_table[['label']]

# COMMAND ----------

# Spliting data into traing set and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify = y)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Modeling
# MAGIC ### There are 3 algorithms that have been tested for matching 
# MAGIC ### 1. Select the candidate with the maximal score as a match

# COMMAND ----------

X_train.loc[:, 'max_scores_sum'] = X_train.groupby(['BSD_Client_ID__c'])['scores_sum'].transform(max)
y_train_pred = (X_train['scores_sum'] == X_train['max_scores_sum']).astype(int)

X_test.loc[:,'max_scores_sum'] = X_test.groupby(['BSD_Client_ID__c'])['scores_sum'].transform(max)
y_test_pred = (X_test['scores_sum'] == X_test['max_scores_sum']).astype(int)

# COMMAND ----------

print(classification_report(y_train, y_train_pred))

# COMMAND ----------

print(classification_report(y_test, y_test_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Set up a threshold, if the scores summation is over the threshold, then the pair is considered as a match

# COMMAND ----------

# Normalize the scores summation to (0,1)
train_scores_sum_normalized = X_train['scores_sum']/5
test_scores_sum_normalized = X_test['scores_sum']/5
best_threshold = getCutoff(train_scores_sum_normalized, y_train, step_size = 0.01)

# COMMAND ----------

y_train_pred = train_scores_sum_normalized.apply(lambda x: 0 if x < best_threshold else 1)
y_test_pred = test_scores_sum_normalized.apply(lambda x: 0 if x < best_threshold else 1)

# COMMAND ----------

print(classification_report(y_train, y_train_pred))

# COMMAND ----------

print(classification_report(y_test, y_test_pred))

# COMMAND ----------

(balanced_matching_table[balanced_matching_table['label'] == 0]['scores_sum']/5).hist()
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC # 3. Modeling: Using simple machine learning techniques to build models (logistic regression and decision tree)

# COMMAND ----------

# Logistic regression
LR_clf = LogisticRegression().fit(X_train[['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score']], y_train)

# COMMAND ----------

y_train_pred = LR_clf.predict(X_train[['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score']])
y_test_pred = LR_clf.predict(X_test[['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score']])

# COMMAND ----------

# Show coefficients
print(LR_clf.coef_)

# COMMAND ----------

print(classification_report(y_train, y_train_pred))

# COMMAND ----------

print(classification_report(y_test, y_test_pred))

# COMMAND ----------

# Decision Tree
DT_clf = DecisionTreeClassifier(max_depth = 3).fit(X_train[['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score']], y_train)

# COMMAND ----------

y_train_pred = DT_clf.predict(X_train[['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score']])
y_test_pred = DT_clf.predict(X_test[['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score']])

# COMMAND ----------

print(classification_report(y_train, y_train_pred))

# COMMAND ----------

print(classification_report(y_test, y_test_pred))

# COMMAND ----------

# Plot tree
plt.figure(figsize=(300,100))
plot_tree(DT_clf, feature_names = ['name_score', 'street_score', 'city_score', 'phone_score', 'postalcode_score'] , filled=True)
display()