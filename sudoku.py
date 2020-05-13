# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 07:52:33 2019

Main Python Script (Driver)

@author: Raj Kishore Patra
"""
import keras
import numpy as np
from model import train_mod
from data_preprocessing import get_data
import sudoku_generator as sg


# x_train, x_test, y_train, y_test = get_data('sudoku.csv')
# train_model(x_train, y_train)

print('Loading the Model...\n')
model = keras.models.load_model('sudoku.model')

def norm(x):   
    return (x/9)-.5

def denorm(x):    
    return (x+.5)*9

def sudoku_game(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    quiz = sample
    
    while(1):
    
        output = model.predict(quiz.reshape((1,9,9,1)))  
        output = output.squeeze()

        prediction = np.argmax(output, axis=1).reshape((9,9))+1 
        probability = np.around(np.max(output, axis=1).reshape((9,9)), 2) 
        
        quiz = denorm(quiz).reshape((9,9))
        breaker = (quiz==0)
     
        if(breaker.sum()==0):
            break
            
        prob_new = probability*breaker
    
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = prediction[x][y]
        quiz[x][y] = val
        quiz = norm(quiz)
    
    return prediction



def test_accuracy(quizes, labels):
    
    counter = 0
    
    for i,quiz in enumerate(quizes):
        prediction = sudoku_game(quiz)
        correct = labels[i].reshape((9,9))+1
        if(abs(correct - prediction).sum()==0):
            counter += 1
            
    print(counter/quizes.shape[0])


#test_accuracy(x_test[:100], y_test[:100])


#game = '080032001703080002500007030050001970600709008047200050020600009800090305300820010'
# sample games
#game = '060400092025000400740239000596020001400005070287040000070802015800507300050004000'

game = sg.main('Easy')

#game = np.array([int(j) for j in game]).reshape((9,9,1))

game_ = game.squeeze()

print('\nGiven puzzle:\n')
print(game_)

game = norm(game)
game = sudoku_game(game)

print('\nSolved puzzle:\n')
print(game)

print('\n')
print(np.sum(game))

