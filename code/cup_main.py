#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:39:30 2020

@author: guido
"""


import os, time, json
import import_dataset 
from neural_network import neural_network
import pandas as pd

start_time = time.time()

### Importa i Dati ###
PATH = os.getcwd()
trCup, blind_test_cup = import_dataset.load_cup(PATH)




### CUP ###
hyp_cup = {"method" : ["batch","minibatch","online"],
                        "hidden_units" : [2, 4, 8,11, 16, 32, [2,5],[2,3,2]],
                       "learning_rates" : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                       "init_method": ["xavier","random"],
                       "activation_function": ["sigmoid","tanh"],
                       "lambda" : [0, 0.001, 0.01, 0.1, 1, 10],
                       "alpha" : [0, 0.2, 0.4, 0.6, 0.8, 0.9]
                       }
### Inizializzazione della rete neurale e import dei dati
cup = neural_network("regression")
cup.load_dataset(trCup.iloc[:,:20],trCup.iloc[:,20:23])
cup.normalize_internal_data("minmax")
cup.split_dataset()


### Prima coarse grid-search su tutti gli iperparametri ad esclusione di momentum e regolarizzazione ###
first_coarse = cup.grid_search(hyp_cup["method"], hyp_cup["hidden_units"], hyp_cup["learning_rates"], init_methods = hyp_cup["init_method"], act_functs = hyp_cup["activation_function"], batch_size = 5) 
pd.DataFrame(first_coarse).to_csv(PATH + "/results/first_coarse_grid_search_cup_1.csv")


### Seconda coarse grid-search su tutti gli iperparametri ad esclusione di momentum e regolarizzazione ###
method_1 = ["online"]
learning_rates_1 = [0.005,0.0075,0.01,0.025,0.05]
hidden_units_1 = [9,10,11,12,13,14,15,16,17,18]
init_method = ["random"]
activation_function = ["sigmoid"]

second_coarse_online = cup.grid_search(method_1, hidden_units_1, learning_rates_1, init_methods = init_method, act_functs = activation_function) 
pd.DataFrame(second_coarse_online).to_csv(PATH + "/results/second_coarse_online_cup_1.csv")


### Terza  grid-search su tutti gli iperparametri###
method_1 = ["online"]
learning_rates_1 = [0.01] 
learning_rates_2 = [0.01]
learning_rates_3 = [0.01]
hidden_units_1 = [10]
hidden_units_2 = [16]
hidden_units_3 = [14]

finer_online_cup_1 = cup.grid_search(method_1, hidden_units_1, learning_rates_1, init_methods = init_method, act_functs = activation_function, lambdas = hyp_cup["lambda"], alphas = hyp_cup["alpha"]) 
finer_online_cup_2 = cup.grid_search(method_1, hidden_units_2, learning_rates_2, init_methods = init_method, act_functs = activation_function, lambdas = hyp_cup["lambda"], alphas = hyp_cup["alpha"]) 
finer_online_cup_3 = cup.grid_search(method_1, hidden_units_3, learning_rates_3, init_methods = init_method, act_functs = activation_function, lambdas = hyp_cup["lambda"], alphas = hyp_cup["alpha"]) 

pd.DataFrame(finer_online_cup_1).to_csv(PATH + "/results/finer_online_cup_1_1.csv")
pd.DataFrame(finer_online_cup_2).to_csv(PATH + "/results/finer_online_cup_2_1.csv")
pd.DataFrame(finer_online_cup_3).to_csv(PATH + "/results/finer_online_cup_3_1.csv")


### Cross validation eseguita con i migliori iperparametri trovati dalle precedenti grid search ##

cross_validation_1 = cup.grid_search(["batch"], [10], [0.01], init_methods = init_method, act_functs = activation_function, lambdas = [0], alphas = [0.2], cross_val = True, k=5)
error_first_cross = cup.error_cross_validation
cross_validation_2 = cup.grid_search(method_1, [16], [0.01], init_methods = init_method, act_functs = activation_function, lambdas = [0], alphas = [0.4], cross_val = True, k=5)
error_second_cross = cup.error_cross_validation
cross_validation_3 = cup.grid_search(method_1, [14], [0.01], init_methods = init_method, act_functs = activation_function, lambdas = [0], alphas = [0.4], cross_val = True, k=5)
error_third_cross = cup.error_cross_validation


pd.DataFrame(cross_validation_1).to_csv(PATH + "/results/cross_validation_1.csv")
pd.DataFrame(cross_validation_2).to_csv(PATH + "/results/cross_validation_2.csv")
pd.DataFrame(cross_validation_3).to_csv(PATH + "/results/cross_validation_3.csv")


### Final Training ###
start_time = time.time()

cup.weight_cleaning()
cup.weight_initialization("random", 14)
cup.training("online", "mse", "sigmoid", learning_rate = 0.01, alpha=0.4, development = True, early_stopping = True, )
cup.plot("error","cup")
mee_final_training_cup = cup.mean_total_error[-1]
mee_final_test_cup = cup.test_error[-1]
print("--- %s seconds ---" % (time.time() - start_time))

final_results_monk1 = { "MEE_training" : mee_final_training_cup,
                       "MEE_test" : mee_final_test_cup,

    }
csv_file = PATH +"/results/Final_results_cup_1.csv"
with open(csv_file, 'w') as file:
     file.write(json.dumps(final_results_monk1))


### Predizione dul blid test set ###

prediction = cup.prediction(blind_test_cup,True)
results = pd.DataFrame(prediction.T)
results.index += 1
results.index
results.to_csv(PATH+ "/results/blind_test_set_prediction.csv")

print("--- %s seconds ---" % (time.time() - start_time))
