#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:30:18 2020

@author: Progetto G
"""


import os, time, json
import import_dataset 
from neural_network import neural_network

### Importa i Dati ###
PATH = os.getcwd()
trmonk_1, tsmonk_1 = import_dataset.load_monk(PATH, 1)
trmonk_2, tsmonk_2 = import_dataset.load_monk(PATH, 2)
trmonk_3, tsmonk_3 = import_dataset.load_monk(PATH, 3)
start_time = time.time()

### MONK ###
hyp_monk = {"method" : ["batch","minibatch","online"],
                        "hidden_units" : [2, 4, 8, 16, 32],
                       "learning_rates" : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                       "init_method": ["xavier","random"],
                       "activation_function": ["sigmoid","tanh"],
                       "lambda" : [0, 0.001, 0.01, 0.1, 1, 10],
                       "alpha" : [0, 0.2, 0.4, 0.6, 0.8, 0.9]
                       }

### inizializzo le reti neurale per la classificazione sul MONK1 ###

monk_1 = neural_network(("classification"))
monk_1.load_dataset(trmonk_1.iloc[:,1:7], trmonk_1.iloc[:,:1])   
monk_1.load_test(tsmonk_1.iloc[:,1:7], tsmonk_1.iloc[:,:1])
monk_1.one_hot_encoding([0,1,2,3,4,5])
monk_1.split_dataset()
best_results_monk1 = monk_1.grid_search(["batch"], [2,3,4,5], [0.1,1,3,10], hyp_monk["lambda"], hyp_monk["alpha"])

###         Best results            ###
###     ['batch', 4, 10, 0, 0.9]    ###

monk_1.weight_cleaning()
monk_1.weight_initialization("random", 4)
monk_1.training("batch", "mse", "sigmoid",learning_rate = 10, alpha= 0.9 ,epoch = 500, development = True, early_stopping = False)
monk_1.plot("accuracy", "monk_1")
monk_1.plot("error", "monk_1")

mse_final_training_monk_1 = monk_1.mean_total_error[-1]
mse_final_test_monk_1 = monk_1.test_error[-1]
accuracy_final_test_monk_1 = monk_1.accuracy_test[-1]
accuracy_final_training_monk_1 = monk_1.accuracy_training[-1]
final_results_monk_1 = { "MSE_training" : mse_final_training_monk_1,
                       "MSE_test" : mse_final_test_monk_1,
                       "Accuracy_training": accuracy_final_training_monk_1,
                       "Accuracy_test": accuracy_final_test_monk_1
    }
csv_file = PATH +"/results/Final_results_monk_1.csv"
with open(csv_file, 'w') as file:
     file.write(json.dumps(final_results_monk_1))


### MONK 2 ###

monk_2 = neural_network(("classification"))
monk_2.load_dataset(trmonk_2.iloc[:,1:7], trmonk_2.iloc[:,:1])   
monk_2.load_test(tsmonk_2.iloc[:,1:7], tsmonk_2.iloc[:,:1])
monk_2.one_hot_encoding([0,1,2,3,4,5])
monk_2.split_dataset()
best_results_monk2 = monk_2.grid_search(["batch"], [2,3,4,5], [0.1,1,3,10], hyp_monk["lambda"], hyp_monk["alpha"])


###         Best results            ###
###     ['batch', 3, 10, 0, 0.9]    ###


monk_2.weight_cleaning()
monk_2.weight_initialization("random", 3)
monk_2.training("batch", "mse", "sigmoid",learning_rate = 10, alpha= 0.9 ,development = True,epoch=500, early_stopping = False)
monk_2.plot("accuracy", "monk_2")
monk_2.plot("error", "monk_2")

mse_final_training_monk_2 = monk_2.mean_total_error[-1]
mse_final_test_monk_2 = monk_2.test_error[-1]
accuracy_final_test_monk_2 = monk_2.accuracy_test[-1]
accuracy_final_training_monk_2 = monk_2.accuracy_training[-1]
final_results_monk_2 = { "MSE_training" : mse_final_training_monk_2,
                       "MSE_test" : mse_final_test_monk_2,
                       "Accuracy_training": accuracy_final_training_monk_2,
                       "Accuracy_test": accuracy_final_test_monk_2
    }
csv_file = PATH +"/results/Final_results_monk_2.csv"
with open(csv_file, 'w') as file:
     file.write(json.dumps(final_results_monk_2))



### MONK 3 ###

monk_3 = neural_network(("classification"))
monk_3.load_dataset(trmonk_3.iloc[:,1:7], trmonk_3.iloc[:,:1])   
monk_3.load_test(tsmonk_3.iloc[:,1:7], tsmonk_3.iloc[:,:1])
monk_3.one_hot_encoding([0,1,2,3,4,5])
monk_3.split_dataset()
best_results_monk3 = monk_3.grid_search(["batch"], [2,3,4,5], [0.1,1,3,10], hyp_monk["lambda"], hyp_monk["alpha"])



###         Best results            ###
###     ['batch', 3, 10, 0, 0.9]    ###
###     ['batch', 3, 10, 0.001, 0.9]    ###


monk_3.weight_cleaning()
monk_3.weight_initialization("random", 3)
monk_3.training("batch", "mse", "sigmoid",learning_rate = 10, alpha= 0.9 ,epoch = 500, development = True, early_stopping = False)
monk_3.plot("accuracy", "monk_3")
monk_3.plot("error", "monk_3")

mse_final_training_monk_3 = monk_3.mean_total_error[-1]
mse_final_test_monk_3 = monk_3.test_error[-1]
accuracy_final_test_monk_3 = monk_3.accuracy_test[-1]
accuracy_final_training_monk_3 = monk_3.accuracy_training[-1]
final_results_monk_3 = { "MSE_training" : mse_final_training_monk_3,
                       "MSE_test" : mse_final_test_monk_3,
                       "Accuracy_training": accuracy_final_training_monk_3,
                       "Accuracy_test": accuracy_final_test_monk_3
    }
csv_file = PATH +"/results/Final_results_monk_3.csv"
with open(csv_file, 'w') as file:
     file.write(json.dumps(final_results_monk_3))


### MONK 3 con regolarizzazione ###
     
monk_3.weight_cleaning()
monk_3.weight_initialization("random", 4)
monk_3.training("batch", "mse", "sigmoid",learning_rate = 10,lamb = 0.001, alpha= 0.9 ,epoch = 500, development = True, early_stopping = False)
monk_3.plot("accuracy", "monk_3_1")
monk_3.plot("error", "monk_3_1")


mse_final_training_monk_3_1 = monk_3.mean_total_error[-1]
mse_final_test_monk_3_1 = monk_3.test_error[-1]
accuracy_final_test_monk_3_1 = monk_3.accuracy_test[-1]
accuracy_final_training_monk_3_1 = monk_3.accuracy_training[-1]
final_results_monk_3_1 = { "MSE_training" : mse_final_training_monk_3_1,
                       "MSE_test" : mse_final_test_monk_3_1,
                       "Accuracy_training": accuracy_final_training_monk_3_1,
                       "Accuracy_test": accuracy_final_test_monk_3_1
    }
csv_file = PATH +"/results/Final_results_monk_3_1.csv"
with open(csv_file, 'w') as file:
     file.write(json.dumps(final_results_monk_3_1))