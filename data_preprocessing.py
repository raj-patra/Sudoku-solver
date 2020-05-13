# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:49:56 2019

Pre Processing Sudoku data

@author: Raj Kishore Patra
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(file):
    
    print('*****Processing the data*****\n')
    
    data = pd.read_csv(file)
    
    quiz_raw = data['quizzes']
    label_raw = data['solutions']
    
    quiz = []; label = [];
    
    for i in quiz_raw:
        
        temp = np.array([int(j) for j in i]).reshape((9,9,1))
        quiz.append(temp)
        
    quiz = np.array(quiz)
    quiz = (quiz/9)-0.5 # Data Normalization
    
    for i in label_raw:
        
        temp = np.array([int(j) for j in i]).reshape((81,1)) -1
        label.append(temp)
        
    label = np.array(label)
    
    del(quiz_raw)
    del(label_raw)
    
    x_train, x_test, y_train, y_test = train_test_split(quiz, label, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test