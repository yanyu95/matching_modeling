# Databricks notebook source
import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz

# COMMAND ----------

def text_preprocessing(df, target_columns, replace_dic = None, remove_list = None):
  """
  Process text data, including turning into lowercase, removing punctuation, turn multiple whitespaces into single whitespace, 
  replacing items based on a customized dictionary and removing items based on a customized list. 
  
  
  Input:
  - df: a pandas dataframe
  - target_columns: A list that contains the names of columns that need to be processed.
  - replace_dic: a dictionary that includes items that need to be replaced and thier replacements
  - remove_list: a list that includes items that need removal
  
  Output:
  pandas dataframe which added preprocessing columns
  """
  # Input data type has to be Pandas Dataframe
  assert(type(df) == pd.core.frame.DataFrame)
  df_processed = df.copy()
  for column in target_columns:
    # Turn to Lowercase
    processed_column = df_processed[column].str.lower()
    # Remove punctuation 
    processed_column = processed_column.str.replace(r'[^\w\s]+', '')
    # Turn multiple whitespaces into single whitespace
    processed_column = processed_column.apply(lambda x: " ".join(x.split()) if x else None)
    # Replace items based on a customized dictionary
    if replace_dic:
      processed_column = processed_column.replace(replace_dic, regex=True)
    # Remove items based on a customized list
    if remove_list:
      processed_column = processed_column.str.split(' ').apply(lambda x : [word for word in x if not word in remove_list])
      processed_column = processed_column.apply(lambda x : ' '.join(x))
    df_processed[column + '_clean'] = processed_column
  return df_processed

# COMMAND ----------

def number_preprocessing(df, target_columns, lenght_limit = None, cut_location= 'last', remove_list = None, remove_consistent_number = False):
  
  """
  Process number, including removing punctuation, single space and abnormal number and limiting length.
  
  Input:
  - df: a pandas dataframe.
  - target_columns: A list that contains the names of columns that need to be processed.
  - length_limit: Limit the length of number n, this function only keep the first/last n digits. Default is keeping all the digits
  - cut_location: indicate that either first or last digits should be kept
  - remove_list: a list that includes items that need removal
  - remove_consistent_number: remove consistent number for example 0000000000
  
  Output:
  pandas dataframe which includes preprocessing columns
  """
  
  # Input data type has to be Pandas Dataframe
  assert(type(df) == pd.core.frame.DataFrame)
  assert(cut_location in ['first', 'last'])
  df_processed = df.copy()
  for column in target_columns:
    processed_column = df_processed[column].apply(lambda x: str(x) if x else None)
    # Remove punctuation 
    processed_column = processed_column.str.replace(r'[^\w\s]+', '')
    # Remove multiple whitespaces 
    processed_column = processed_column.apply(lambda x: "".join(x.split()) if x else None)
    if lenght_limit:
      if cut_location == 'last':
        # Keep the last n digits
        processed_column = processed_column.str[-lenght_limit:]
      elif cut_location == 'first':
        processed_column = processed_column.str[:lenght_limit]
    # Remove number based on a customized list
    if remove_list:
      processed_column = processed_column.apply(lambda x : None if x in remove_list else x)
    # turn the consistent number for example 0000000000 into null
    if remove_consistent_number:
      processed_column = processed_column.map(lambda x: None if len(set(x)) == 1 else x, na_action='ignore')
    df_processed[column + '_clean'] = processed_column
  return df_processed

# COMMAND ----------

# Basic cleaning dictionary/list for preprocessing functions. specially for company name and address
basic_name_replace_dic = {
  # Replace company with co, and corporation with corp
  'company':'co',
  'corporation':'corp'
}

# Company suffixes
suffixes = ['inc', 'llc', 'ltd', 'lp', 'co', 'service', 'services', 'corp']
# Stop Word
stop_word = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
basic_name_remove_list = suffixes + stop_word


basic_address_replace_dic = { 
  # Address converts to abbreviation
  'alley' : 'aly',
  'avenue' : 'ave',
  'boulevard' : 'blvd',
  'circle' : 'cir',
  'court' : 'ct',
  'cove' : 'cv',
  'canyon' : 'cyn',
  'drive' : 'dr',
  'expressway' : 'expy',
  'highway' : 'hwy',
  'lane' : 'ln',
  'parkway' : 'pkwy',
  'place' : 'pl',
  'point' : 'pt',
  'road' : 'rd',
  'square' : 'sq',
  'street' : 'st',
  'terrace' : 'ter',
  'way' : 'wy', 
  # Direction converts to abbreviation
  'south' : 's',
  'north' : 'n',
  'east' : 'e',
  'west' : 'w',
  'northwest': 'nw',
  'northeast': 'ne',
  'southwest': 'sw',
  'southeast': 'se'
}

# COMMAND ----------

def scores_assignment(df, fuzzy_columns = None, validation_columns = None):

  '''
  Assigned scored to the paired columns. The columns can be socred by 2 methods: fuzzy matching supported by fuzzywuzzy package and excat matching
  
  Input:
  - target_table: A pandas dataframe that has the pairs need to be scored
  - fuzzy_columns: Indicates the columns that need to be scored by fuzzy matching. The type is dictionary and the format is {'column':['column_1','column_2', scorer]}. 'column_1'and 'column_2'
                   is the names of paird columns. scorer is fuzzy matching functions. The options are fuzz.ratio, fuzz.partial_ratio, fuzz.ratio, fuzz.token_sort_ratio and fuzz.token_set_ratio. 
                   The range of score is from 0 to 100
  - validation_columns: Indicates the columns that need to be scored by excat matching. If the values are excatly the same, the score is 1, otherwise 0. The type is dictionary and the format is
                  {'column':['column_1','column_2']}
  
  Output:
  A pandas dataframe attached by score columns.
  '''
  
  df_scores = df.copy()
  if fuzzy_columns:
      for key in fuzzy_columns.keys():
          column_name = '{}_score'.format(key)
          df_scores[column_name] = df_scores.apply(lambda x: fuzzy_columns[key][2](x[fuzzy_columns[key][0]], x[fuzzy_columns[key][1]]), axis=1)

  if validation_columns:
      for key in validation_columns.keys():
          column_name = '{}_score'.format(key)
          df_scores[column_name] = (df_scores[validation_columns[key][0]] == df_scores[validation_columns[key][1]]).astype(int)

  return df_scores
