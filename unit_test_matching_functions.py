# Databricks notebook source
import unittest
import pandas as pd
import string
from matching_library import *

# COMMAND ----------

df_needs_preprocessing = pd.DataFrame({'text': ['UPPERCASE', 'p.u.n.c.t.u.a.t.i.o.n', 'multiple   spaces', 'apple needs replacement', 'suffix needs removal'], 
                            'number': ['+12345', 123456, 54321, 88888, 12345]})
df_needs_matching_scores = pd.DataFrame({'name_1': ['ajg'], 'name_2': ['ajg company'],
                            'phone_1':['123456'],'phone_2':['123456'] })

lenght_limit = 5

class matching_functions_test(unittest.TestCase):
 
  def test_text_preprocessing(self, test_input = df_needs_preprocessing): 
    '''
    Test whether the output of text preprocessing function is as expectation
    
    Input:
    test input - pandas data frame that contains the columns that need to be processed.
    '''
    
    actual_output = text_preprocessing(test_input, target_columns = ['text'], replace_dic = {'apple':'banana'}, remove_list = ['suffix'])
    self.assertTrue(any(actual_output['text_clean'].str.isupper()) == False), "Test whether the text was turned to Lowercase"
    self.assertTrue(any(actual_output['text_clean'].apply(lambda x: any(char in x for char in string.punctuation))) == False),  "Test whether punctuations were removed"
    self.assertTrue(any(actual_output['text_clean'].apply(lambda x: (len(x.split())-1) != x.count(' '))) == False),  "Test whether multiple whitespaces were removed"
    self.assertTrue(any(actual_output['text_clean'].apply(lambda x: 'apple' in x)) == False),  "Test whether the keys in replace_dic were removed"
    self.assertTrue(any(actual_output['text_clean'].apply(lambda x: 'suffix' in x)) == False),  "Test whether the items in remove_listc were removed"
    
  def test_number_preprocessing(self, test_input = df_needs_preprocessing):
    '''
    Test whether the output of text preprocessing function is as expectation
    
    Input:
    test input - pandas data frame that contains the columns that need to be processed.
    '''
    actual_output = number_preprocessing(test_input, target_columns = ['number'], lenght_limit = lenght_limit, remove_list = ['54321'], cut_location = 'last', remove_consistent_number = True)
    
    self.assertTrue(any(actual_output[actual_output['number_clean'].notnull()].apply(lambda x: any(char in x['number_clean'] for char in string.punctuation), axis = 1)) == False),  "Test whether punctuations were removed"
    self.assertTrue(any(actual_output[actual_output['number_clean'].notnull()].apply(lambda x: len(x['number_clean']) != lenght_limit, axis = 1)) == False),  "Test whether the length of number is correct"
    self.assertTrue(any(actual_output[actual_output['number_clean'].notnull()].apply(lambda x: '54321' in x)) == False),  "Test whether the items in remove_listc were removed"
    self.assertTrue(any(actual_output[actual_output['number_clean'].notnull()].apply(lambda x: len(set(x['number_clean'])) == 1, axis = 1)) == False),  "Test whether consistant number were removed"
    
  def test_scores_assignment(self, test_input = df_needs_matching_scores):
    '''
    Testing scores assignment functions is working
    
    Input:
    test_input - pandas data frame that has the pairs need to be scored
    '''

    actual_output = scores_assignment(test_input, fuzzy_columns = {'name':['name_1', 'name_2', fuzz.token_sort_ratio]}, validation_columns = {'phone':['phone_1', 'phone_2', fuzz.token_sort_ratio]})
    test_output = test_input.copy()
    test_output['name_score'] = test_input.apply(lambda x: fuzz.token_sort_ratio(x['name_1'], x['name_2']), axis=1)
    test_output['phone_score'] = (test_input['phone_1'] == test_input['phone_2']).astype(int)
    
    self.assertEqual(len(actual_output.columns), 4+2), "Test whether the new score columns were added"
    self.assertTrue(actual_output.equals(test_output)), "Test whether the scores were calculated correctly"

# COMMAND ----------

suite = unittest.TestLoader().loadTestsFromTestCase(matching_functions_test)
runner = unittest.TextTestRunner(verbosity = 2)
runner.run(suite)
