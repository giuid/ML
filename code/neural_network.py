#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:29:46 2020

@author: guido & giuseppe
"""
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os,math
PATH = os.getcwd()
np.seterr(all='ignore')


"""
Definiamo la classe neural_network che ci permetterà di creare le reti neurali artificiali in base ai 
dati di input e ai parametri scelti
"""
class neural_network:
    
    
    """
    Inizializiamo la classe prendendo in input solamente il tipo di task, classificazione o regressione.
    Inizializiamo le variabili globali utilizzate nei metodi successivi
    """
    def __init__(self, task):
        self.task = task
        assert self.task in ["classification",
                             "regression"], "The available tasks are 'classification' and 'regression'"
        self.dataset_input = None
        self.dataset_output = None
        self.training_input = None
        self.training_output = None
        self.validation_input = None
        self.validation_output = None
        self.development_input = None
        self.development_output = None
        self.test_input = None
        self.test_output = None
        self.test_flag = False
        self.weight_flag = False
        self.normalization_flag = False
        self.normalization_method = None
        self.weights_method = ""
        self.weights = []
        self.bias = []
        self.hidden_layers = None
        self.original_input = None
        self.original_output = None
        self.hidden_nodes = None
        self.output_nodes = None
        self.mean_total_error = []
        self.activation_function_layer = []
        self.learning_rate = None
        self.momentum = None
        self.regularization = None
        self.loss_function = None
        self.activation_function = None
        self.deep_search = False
        self.accuracy_training = []
        self.accuracy_validation = []
        self.validation_error = []
        self.layers = []
        self.development = False
        self.grid_search_flag = False
        self.test_error = []
        self.accuracy_test = [] 
        self.epoch = []
        self.error_cross_validation = []
   
    
    """
    Creiamo un metodo da richiamare per effettuare un controllo sui dati all'interno dei vari metodi
    """    
    def __data_check(self, method, var=None, var2=None):
        assert method in ["data", "normalization", "initialization", "weightini", "activation", "loss"]
        if method == "data":
            assert self.dataset_input.size > 0, "You should load some data before calling any function. Use the method " \
                                                 "load_training() "
            assert self.dataset_output.size > 0, "You should load some data before calling any function. Use the method " \
                                                  "load_training() "
        elif method == "normalization":
            assert self.normalization_flag == False, "You already normalized the input,  if you want to restore the original data you should call the method restore_internal_data()"
            assert var in ["simple", "minmax",
                           "z-score"], "The available scaling methods are 'simple','minmax','z-score'"

        elif method == "initialization":
            assert var in ["xavier", "he",
                           "typeone",
                           "random"], "The available methods for weights initialization are 'xavier', 'he','typeone'"

        elif method == "weightini":
            assert self.weight_flag == False, "you already initialized the weights with " + self.weights_method + " method. If you want to initialize the Weights againg you should call the method 'weight_cleaning'"
            assert var in ["xavier", "he",
                           "typeone",
                           "random"], "The available methods for weights initialization are 'xavier', 'he','typeone','random'"
            assert type(self.dataset_input.shape[
                            1]) == int, "the variable 'input_nodes' should be and integer, before you intialize the weights you should load a Data set"
            if type(var2) != int:
                assert len(var2) in [1, 2, 3, 4,
                                     5], "You should specify a numer of hidden units, the function take in input a vector [nodesfor1stlayer,nodesfor2ndlayer,...] with a maximum of 5 layers"

            assert type(self.output_nodes) == int, "the variable 'output_nodes' should be and integer"

        elif method == "activation":
            assert var in ["sigmoid", "tanh", "relu",
                           "linear"], "You can use the following functions: 'sigmoid', 'tanh', 'relu','linear'"
        elif method == "loss":
            assert var in ["mse", "mae", "mee"], "You can use the following functions: 'mse', 'mae', 'mee'"

    
    
    """
    Creiamo un metodo per la descrizione dei dati che verranno utilizzati come Training
    N.B. le variabili interne alla classe self.dataset_input e self.dataset_output sono quelle che subiranno tutte le modifiche necessarie
    per il corretto funzionamento della rete, per il training set originale si faccia riferimento alle variabili self.original_input e self.original_output
    """
    def data_description(self):
        self.__data_check("data")
        print("Right now the input dataset is described as follow: ")
        print(pd.DataFrame(self.dataset_input).describe())
        print("Right now the out dataset is described as follow: ")
        print(pd.DataFrame(self.dataset_output).describe())

 
    
    
    """
    Creiamo il metodo load_training che prende in input il training set già splittato in input e output.
    La funzione controlla se è stata selezionata la regressione o la classificazione.
    In entrambi i casi controlla i nodi in output da creare e li inserisce in una variabile globale
    """
    def load_dataset(self, dataset_input, dataset_output): 
        self.dataset_input = np.array(dataset_input)
        self.dataset_output = np.array(dataset_output)
        self.original_input = np.array(dataset_input)
        self.original_output = np.array(dataset_output)
        if self.task == "regression" and len(self.dataset_output.shape) > 1:
            self.output_nodes = self.dataset_output.shape[1]
        elif self.task == "regression" :
            self.output_nodes = 1
        else:
            assert dataset_output.shape[1] == 1 or len(self.dataset_output.shape) == 1, "For the classification task you should use just one column output"
            if len(np.unique(dataset_output)) > 2:
                self.output_nodes = len(np.unique(dataset_output))
            else:
                self.output_nodes = 1
    
    
    
   
    """
    Creiamo il metodo per importare un test-set esterno
    """
    def load_test(self,test_input, test_output):
        self.test_flag = True
        self.test_input = np.array(test_input)
        self.test_output = np.array(test_output)
    
    
        
     
    """
    Metodo per la creazione dei subset necessari all'addestramento della rete naurale artificiale
    """
    def split_dataset(self):
        dataset_input = pd.DataFrame(self.dataset_input)
        dataset_output = pd.DataFrame(self.dataset_output)
        development_set = pd.concat([dataset_input,dataset_output], axis=1)
        if not self.test_flag:
            test_set = development_set.sample(frac= 1/4)
            development_set = development_set.drop(test_set.index)
            test_set = np.array(test_set)
            self.test_input = test_set[:,:self.dataset_input.shape[1]]
            self.test_output = test_set[:, self.dataset_input.shape[1]:]
        dev_set = pd.DataFrame(development_set)
        mini_validation_set = dev_set.sample(frac=1/10)
        dev_set = dev_set.drop(mini_validation_set.index)
        mini_validation_set = np.array(mini_validation_set)
        dev_set = np.array(dev_set)
        validation_set = development_set.sample(frac = 1/3)
        training_set = development_set.drop(validation_set.index)
        training_set = np.array(training_set)
        validation_set = np.array(validation_set)
        self.development_input = dev_set[:, :self.dataset_input.shape[1]]
        self.development_output = dev_set[:, self.dataset_input.shape[1]:]
        self.training_input = training_set[:,:self.dataset_input.shape[1]]
        self.training_output = training_set[:, self.dataset_input.shape[1]:]
        self.validation_input = validation_set[:,:self.dataset_input.shape[1]]
        self.validation_output = validation_set[:, self.dataset_input.shape[1]:]
        self.mini_validation_input = mini_validation_set[:,:self.dataset_input.shape[1]]
        self.mini_validation_output = mini_validation_set[:, self.dataset_input.shape[1]:]
        


    """
    Creiamo il metodo One-Hot-Encoding che prende in  input una stringa o un vettore di stringhe contenente
    i nomi delle colonne alle quali applicare tale codifica. Per fare ciò viene utilizzato il comodo metodo get_dummies
    della libreria pandas
    """    
    def one_hot_encoding(self, columns):
        self.__data_check("data")
        if type(columns) == list:
            for el in columns:
                assert type(el) == int, "In the variable 'columns' you should put an integer or an array of integers with the position of the columns (the first column correspond to 0) .\nIf you want to check your data you should call the method 'data_description'"
            df = pd.DataFrame(self.dataset_input)
            for el in columns:
                a = pd.get_dummies(df[el], prefix=el)
                df = df.drop(columns=el)
                df = pd.concat([df, a], axis=1)
            self.dataset_input = np.array(df)
            if self.test_flag:
                df = pd.DataFrame(self.test_input)
                for el in columns:
                    a = pd.get_dummies(df[el], prefix=el)
                    df = df.drop(columns=el)
                    df = pd.concat([df, a], axis=1)
                self.test_input = np.array(df)
                


   
    """
    Di seguito le funzioni per normalizzare i dati in input ed eventualmente i dati in output. 
    La prima normalizza i dati precedentemente caricati, la seconda contiene solo le regole di normalizzazione a seconda del metodo scelto
    I metodi di normalizzazione implementati sono il metodo 'simple' che divide il pattern per il max, il "min/max" e lo "z-score"
    """    
    def normalize_internal_data(self, method, output= False):
        self.__data_check("data")
        self.__data_check("normalization", method)
        self.dataset_input = self.__normalize_data(self.dataset_input, method)
        if output: 
            self.dataset_output =  self.__normalize_data(self.dataset_output, method)
            self.normalization_flag = True

        if self.test_flag:
            self.test_input =  self.__normalize_data(self.test_input, method)
            if output:self.test_output =  self.__normalize_data(self.test_input, method)
        self.normalization_method = method
        print("Normalization of all the training set with '" + method + "' method completed.")
    
    
    def __normalize_data(self, input_data, method):
        input_data = np.array(input_data)
        if method == "simple":
            normalized_data = np.divide(input_data ,np.abs(input_data).max(axis=0))           
        if method == "minmax":
            normalized_data = np.divide( (np.subtract(input_data, (input_data).min(axis=0))) , np.subtract((input_data).max(axis=0), (input_data).min(axis=0)))
        if method == "z-score":
            normalized_data = (input_data - np.mean(input_data, axis=0)) / (np.std(input_data, axis=0))            
        return normalized_data
     
    
    
    
    """    
    Il metodo denormalize permette di ripristinare i dati nella scala originale prima della normalizzazione        
    """    
    def __denormalize(self, output):
        method = self.normalization_method
        if method == "simple":     
            maximum = np.abs(self.original_output).max(axis=0)
            maximum = np.reshape(maximum, (len(maximum),1))
            real_output = np.multiply(output, maximum)
        if method == "minmax":
            minmax = np.subtract((self.original_output).max(axis=0), (self.original_output).min(axis=0))
            minimum = (self.original_output).min(axis=0)
            minmax = np.reshape(minmax, (len(minmax),1))
            minimum = np.reshape(minimum, (len(minimum),1))
            real_output = np.add(np.multiply(output,minmax), minimum)
        if method == "z-score":
            mean = np.mean(self.original_output, axis=0)
            mean = np.reshape(mean,(len(mean),1))
            std = np.std(self.original_output, axis=0)
            std = np.reshape(std,(len(std),1))
            real_output = np.add(np.multiply(output, std),mean)
        return real_output

   
    
    
    """
    Metodo per ripristinare il training set così come era stato caricato originariamente
    """    
    def restore_internal_data(self):
        import copy
        if self.norm_input_flag == True or self.norm_output_flag == True:
            self.dataset_input = copy.copy(self.original_input)
            self.norm_input_flag = False
            self.dataset_output = copy.copy(self.original_output)
            self.norm_output_flag = False
            print("The data was restored as the one you first uploaded")
        else:
            print("You never normalized the data")
            
   
   
    """
    __initialization_method permettere di scegliere il metodo di inizializzazione che si vuole, sarà richiamato in sel.weight_initialization()
    """
    def __initialization_method(self, method, x, y):
        self.__data_check("initialization", method)
        if method == "xavier":
            weights = np.random.randn(x, y) * np.sqrt(1 / y)
        elif method == "he":
            weights = np.random.randn(x, y) * np.sqrt(2 / y)
        elif method == "typeone":
            weights = np.random.randn(x, y) * np.sqrt(2 / (x + y))
        elif method == "random":
            weights = np.random.uniform(low=-0.7, high=0.7, size=(x, y))
        return weights
    
  
    
    """
    Il metodo weight_inizialization prende in input il metodo di inizializzazione e il numero di hidden nodes, accetta anche array di interi
    i valori per ogni indice corrisponderanno ai nodi di un hidden layer:
    e.g. se hiddennodes=[2,3] la rete avrà due layer nascosti, il primo costituito da 2 nodi e il secondo da 3
    I metodi di inizializzazione implementati sono lo 'xavier', lo 'he' , il 'typeone' e il 'random' (valori random in [-0.7, 0.7])
    """
    def weight_initialization(self, method, hiddennodes):
        self.__data_check("data")
        self.__data_check("weightini", method, hiddennodes)
        self.hidden_nodes = hiddennodes
        input_nodes = self.dataset_input.shape[1]
        output_nodes = self.output_nodes
        if type(hiddennodes) != int:
            hidden_layers = len(hiddennodes)
            i = 0
            for el in hiddennodes:
                if i == 0:
                    i += 1
                    w1 = self.__initialization_method(method, el, input_nodes)
                    self.weights.append(w1)
                    b = np.random.randn(el, 1)
                    self.bias.append(b)
                    e = el
                elif (i + 1) < len(hiddennodes):
                    i += 1
                    w1 = self.__initialization_method(method, el, e)
                    self.weights.append(w1)
                    b = np.random.randn(el, 1)
                    self.bias.append(b)
                    e = el
                elif (i + 1) == len(hiddennodes):
                    hh = self.__initialization_method(method, el, e)
                    w1 = self.__initialization_method(method, output_nodes, el)
                    b1 = np.random.randn(el, 1)
                    b = np.random.randn(output_nodes, 1)
                    self.bias.append(b1)
                    self.bias.append(b)
                    self.weights.append(hh)
                    self.weights.append(w1)
        else:
            hidden_layers = 1
            ih = self.__initialization_method(method, hiddennodes, input_nodes)
            ho = self.__initialization_method(method, output_nodes, hiddennodes)
            ihb = np.random.randn(hiddennodes, 1)
            hob = np.random.randn(output_nodes, 1)
            self.weights.append(ih)
            self.weights.append(ho)
            self.bias.append(ihb)
            self.bias.append(hob)
        self.hidden_layers = hidden_layers
        self.weight_flag = True
        self.weights_method = method

        
    
    """
    La funzione weight_cleaning azzera i pesi e i bias riporta le flag a False, permettendo di inizializzare i pesi nuovamente.
    Senza tale funzione non sarà possibile re-inizializzare i pesi
    """
    def weight_cleaning(self):
        self.weight_flag = False
        self.weights = []
        self.bias = []

    


    """
    Il metodo activation permette di scegliere la funzione di attivazione e restituisce
    f(net) oppure la derivata f'(net)
    """  
    def activation(self, method, net, der=False):
        self.__data_check("activation", method)
        if not der:
            if method == "sigmoid":
                return 1.0 / (1 + np.exp(-net))
            elif method == "tanh":
                return (np.exp(net) - np.exp(-net)) / (np.exp(net) + np.exp(-net))
            elif method == "relu":
                return np.maximum(0, net)
            elif method == "linear":
                return net
        else:
            if method == "sigmoid":
                return (1.0 / (1 + np.exp(-net))) * (1 - (1.0 / (1 + np.exp(-net))))
            elif method == "tanh":
                return 1 - np.power((np.exp(net) - np.exp(-net)) / (np.exp(net) + np.exp(-net)), 2)
            elif method == "relu":
                net[net<=0]=0
                net[net>0]=1
                return net 
            elif method == "linear":
                return 1

            
            
            
    """
    Il metodo loss() permette di scegliere la loss function e restituisce l'errore totale o la derivata
    """        
    def loss(self, method, output_layer, actual_layer, derivative=False):
        self.__data_check("loss", method)
        error = np.subtract(output_layer, actual_layer)

        if derivative == False:
            if method == "mse":
                mse = (1 / 2) * ((error ** 2))
                if len(mse.shape) < 2:
                    mse = mse.reshape(len(mse), 1)
                return error, mse
            elif method == "mee":
                mee = (np.sqrt((error ** 2).sum(axis=0)))
                if len(mee.shape) < 2:
                    mse = mee.reshape(len(mee), 1)
                return error, mee
        elif derivative == True:
            if method == "mse":
                return error
            elif method == "mee":
                total_error = (np.sqrt((error ** 2)).sum(axis=0))
                return np.divide(error, total_error)
       
            
    
    """
    Metodo per calcolare la fase di forward, fornisce in output due liste, una contenente le varie netj(n) e l'altra contenente i valori
    dei nodi dei layer = f(netj(n))
    """   
    def __forward(self, inpt):
        activation_function = self.activation_function
        if len(inpt.shape) == 1:
            lay = inpt.reshape(len(inpt),1 )
        else:
            lay = inpt.T
        net = []
        layers = []
        net.append(lay)
        layers.append(lay)

        for el in range(self.hidden_layers + 1):
            weights = self.weights[el]
            bias = self.bias[el]
            lay = np.add(np.dot(weights, lay), bias)
            net.append(lay)
            if self.task == "regression" and el == self.hidden_layers:
                lay = self.activation("linear", lay)
            else:
                lay = self.activation(activation_function, lay)
            layers.append(lay)
        return layers, net
    
    
    
    
    """
    Il metodo update_weights prende in input la derivata del gradiente rispetto ai pesi e al bias per un determinato layer
    e aggiorna l'array dei pesi e dei bias    
    """  
    def __update_weights(self, delta_w, delta_b, position):
        new_weights = self.weights[-(position)] + delta_w
        new_bias = self.bias[-(position)] + delta_b
        self.bias.insert(-(position), new_bias)
        self.bias.pop(-(position))
        self.weights.insert(-(position), new_weights)
        self.weights.pop(-(position))
        
        
        
   
    """
    Il seguente metodo applica le regole di backpropagation dati un input e un output tenendo conto, in maniera automatica,
    del metodo di addestramento utilizzato (batch,online, minibatch)
    """     
    def __backpropagation(self, inp, output, delta_w_old=[], delta_b_old=[]):
        if len(inp.shape) == 1:
            input_lenght = 1
        else:
            input_lenght = inp.shape[0]
        if len(output.shape) == 1:
            output = output.reshape(1, len(output.T))
        if len(delta_w_old) == 0:
            delta_w_old = [0] * (self.hidden_layers + 1)
        if len(delta_b_old) == 0:
            delta_b_old = [0] * (self.hidden_layers + 1)
        ### assegno le variabili di sitema a variabili locali ###
        activation_function = self.activation_function
        loss_function = self.loss_function
        lamb = self.regularization
        learning_rate = self.learning_rate
        alpha = self.momentum
        
        ### chiamo la funzione forward e calcolo i valori delle unità per tutti i layer della rete ###
        layers, net = self.__forward(inp)
        delta_b_current = []
        delta_w_current = []
        output_layer = output.T
        actual_layer = layers[-1]
        
        ### calcolo l'errore grazie al metodo loss per l'output layer ###
        error, loss = self.loss(loss_function, output_layer, actual_layer)
        regularization = (self.weights[-1] ** 2).sum() * lamb
        total_error = loss.sum() + regularization

        if self.task == "regression":
            f1_net = self.activation("linear", net[-1], der=True)
        else:
            f1_net = self.activation(activation_function, net[-1], der=True)
        error_derivative = self.loss(loss_function, output_layer, actual_layer, True)
        output_layer_error = np.multiply(error_derivative, f1_net)
        bias_layer = np.array([1]*input_lenght).reshape(input_lenght,1)
        gradient_b = np.dot(output_layer_error, bias_layer)
        gradient_w = np.dot(output_layer_error, layers[-2].T)
        delta_w = (1 / input_lenght) * learning_rate * gradient_w - 2 * lamb * self.weights[-1] + alpha * delta_w_old[-1]
        delta_b = (1 / input_lenght) * learning_rate * gradient_b - 2*lamb * self.bias[-1] + alpha * delta_b_old[-1]
        self.__update_weights(delta_w, delta_b, (1))
        previous_layer_error = output_layer_error
        delta_w_current.insert(0, delta_w)
        delta_b_current.insert(0, delta_b)
        for i in range(1, len(layers) - 1):
            current_layer_input = layers[-(i + 2)].T
            current_layer_weights = self.weights[-i]
            deltaSum = (np.dot(previous_layer_error.T, current_layer_weights)).T
            f1_net = self.activation(activation_function, net[-(i + 1)], der=True)
            
            ### current_layer_error corrisponde al delta piccolo, cioè il gradiente locale ###
            current_layer_error = np.multiply(f1_net, deltaSum)
            gradient_w = np.dot(current_layer_error, current_layer_input)
            gradient_b = np.dot(current_layer_error, bias_layer)

            delta_w = (1 / input_lenght) * learning_rate * gradient_w - 2 * lamb * self.weights[-(i + 1)] + alpha * delta_w_old[-(i + 1)]
            delta_b = (1 / input_lenght) * learning_rate * gradient_b - 2 * lamb * self.bias[-(i + 1)] + alpha * delta_b_old[-(i + 1)]
            self.__update_weights(delta_w, delta_b, (i+1))
            previous_layer_error = current_layer_error
            delta_w_current.insert(0, delta_w)
            delta_b_current.insert(0, delta_b)
        return total_error, delta_w_current, delta_b_current
    
   
    
    
    
    """
    Il metodo training permette di addestrare il modello: 
    prende in input gli hyperparametri e addestra un modello. 
    I dati in input sono mescolati per ogni epoca.
    Il metodo permette l'esecuzione sia con il batch che l'on-line che il minibatch
    La tecnica di early stopping è questa: se la differenza con l'errore di tante epoche prima (tante quanto la patience) 
    è minore di 0.001 o l'errore riprende a crescere l'algoritmo si ferma (sempre in base alla patience).
    In questa fase vengono richiamati i metodi necessari per calcolare l'errore sul training e sul validation 
    ed eventualmente l'accuracy (per la classificazione) e le statistiche sul test_set (solo nel final training)'
    """
    def training(self, method, loss_function, activation_function,learning_rate=0.01, lamb=0, alpha=0, epoch=2000, batch_size = 1, development = False, early_stopping=False):
        assert self.weights != [], "You have to initialize the weights before you can train a model"
        self.loss_function = loss_function
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.momentum = alpha
        self.regularization = lamb
        self.mean_total_error = []
        self.validation_error = []
        self.test_error = []
        self.accuracy_test = []
        self.development = development
        self.epoch = 0
        mean_total_error = 0
        patience =0
        ni = 0

        if method == "online": patience = 50
        if method == "minibatch": patience = 75
        if method == "batch": patience = 20
        
        if self.task == "classification":
            self.accuracy_training = []
            self.accuracy_validation = []
            
        ### Selezione dei dataset e sottodataset da utilizzare in fase di training ###     
        if development:
            shuffled_data = np.concatenate((self.development_input, self.development_output), axis=1)
            validation_input = self.mini_validation_input
            validation_output = self.mini_validation_output           
        else:
            shuffled_data = np.concatenate((self.training_input, self.training_output), axis=1)
            validation_input = self.validation_input
            validation_output = self.validation_output
            
        if not self.grid_search_flag:self.__print_progress_bar(0, epoch,prefix = 'Progress:', suffix = 'Complete')
        delta_w_old, delta_b_old = [], []
        for i in range(epoch):
           
            ### controlle se deve eseguire l'addestramento sul training set o sul development ###
            ### in ogni caso seleziona i dati appropriati e li mescola stando attenti alla corrispondenza pattern-input / output ###
            if not self.grid_search_flag:self.__print_progress_bar(i, epoch,prefix = 'Progress:', suffix = 'Complete')
            np.random.shuffle(shuffled_data)
            training_input = shuffled_data[:,:self.training_input.shape[1]]
            training_output = shuffled_data[:, self.training_input.shape[1]:]
            
            ### In base al metodo scelto quando la funzione viene invocato compio un addestramento online o batch ###
            if method == "online":
                total_error = 0
                delta_w_old, delta_b_old = [] , []
                for x, y in zip(training_input, training_output):
                    errore, delta_w_old, delta_b_old = self.__backpropagation(x, y, delta_w_old, delta_b_old)
                    total_error += errore
                mean_total_error = total_error / len(training_input)
                
            elif method == "batch":
                total_error, delta_w_old, delta_b_old = self.__backpropagation(training_input, training_output, delta_w_old, delta_b_old)
                mean_total_error = total_error / len(training_input)
                
            elif method == "minibatch":
                if batch_size < 2:
                    print("The Batch size for the minibatch method should be 2 or higher, otherwise you'll use a simple batch")
                split_size = int(training_input.shape[0]/batch_size)
                mean_total_error = 0
                for batch in range(batch_size):
                    total_error, delta_w_old, delta_b_old = self.__backpropagation(training_input[batch*split_size:(batch+1)*split_size], training_output[batch*split_size:(batch+1)*split_size], delta_w_old, delta_b_old)
                    
                    mean_total_error += total_error / split_size
                mean_total_error /= batch_size
            
            self.mean_total_error.append(mean_total_error)
                 
            ### calcolo dell'accuracy per il classification task
            if self.task == "classification" and not self.grid_search_flag:
                predicted_layer_validation = self.prediction(validation_input)
                predicted_layer_training = self.prediction(training_input)
                accuracy_validation = self.__accuracy(predicted_layer_validation, validation_output)
                accuracy_training = self.__accuracy(predicted_layer_training, training_output)
                self.accuracy_training.append(accuracy_training)
                self.accuracy_validation.append(accuracy_validation)
                
            ### calcolo dell'errore medio sul validation set, usa la MEE per la regressione e l' MSE per la classificazione    
            if self.task == "regression":
                val_error, val_mean_total_error = self.evaluate(validation_input,validation_output, "mee")
            elif self.task == "classification":
                val_error, val_mean_total_error = self.evaluate(validation_input,validation_output, "mse")
            self.validation_error.append(val_mean_total_error)
            
            ## durante il final training l'algoritmo calcola l'errore sul test-set per la creazione del grafico ###
            if development:
                if self.task == "regression":
                    test_error, test_mean_total_error = self.evaluate(self.test_input,self.test_output, "mee")
                elif self.task == "classification":    
                    test_error, test_mean_total_error = self.evaluate(self.test_input,self.test_output, "mse")
                    predicted_layer_test = self.prediction(self.test_input)
                    accuracy_test = self.__accuracy(predicted_layer_test, self.test_output)
                    self.accuracy_test.append(accuracy_test)
                self.test_error.append(test_mean_total_error) 
                              
            ##early stopping criteria###  
            if math.isnan(val_mean_total_error) or math.isinf(val_mean_total_error):                    
                self.mean_total_error.pop()
                self.validation_error.pop()
                if len(self.mean_total_error) <=1: 
                    self.mean_total_error.append("error")
                    self.validation_error.append("error")
                self.epoch = i
                break
                
            if early_stopping:
                if i>patience:
                    if val_mean_total_error > min(self.validation_error) or self.validation_error[-patience] - val_mean_total_error < 0.001 :
                        ni += 1
                    if ni == patience or val_mean_total_error - self.validation_error[-patience] > 25 :
                        if not self.grid_search_flag:self.__print_progress_bar(epoch, epoch,prefix = 'Progress:', suffix = 'Complete')

                        self.epoch = i+1
                        break
            if i==epoch-1: 
                self.epoch = i+1
                if not self.grid_search_flag:self.__print_progress_bar(epoch, epoch,prefix = 'Progress:', suffix = 'Complete')


    
    
    """
    il metodo plot stampa il plot dell'errore o dell'accuracy al trascorrere delle epoche
    """      
    def plot(self, method, output_file):
        assert method in ["error","accuracy"], "you should pick a method among these 'error', 'accuracy'"
        error = self.mean_total_error
        epoch = np.arange(1, len(error) + 1, 1)
        #determina la dimensione del plot
        pl.figure(figsize=(10, 7.5))
        #1° Grafico (Blu)
        if method == "error":
            pl.plot(epoch, error, label="Train")
            #2° Grafico (Arancione)
            if self.development:
                pl.plot(epoch, self.test_error, label="Test",linestyle='dashed')
            else:
                pl.plot(epoch, self.validation_error, label="Validation", linestyle='dashed')
            pl.xlabel("Epoch")
            pl.ylabel("Loss")
            pl.title('Curva Apprendimento')
            #inserisce la griglia nel grafico
            pl.grid()
            #consente di salvare il plot in una directory
            pl.savefig(PATH + '/plots/Error_'+output_file+'.png')
            #Il comando plt.legend consente di inserire una legenda nel grafico e determinarne la posizione
            #per i posizionamenti alternativi della legend e per il controllo delle sue ulteriori proprietà consultare help(pyplot.legend)
            pl.legend(loc="upper right") 
        elif method == "accuracy":
            accuracy_training = self.accuracy_training
            pl.plot(epoch, accuracy_training, label="Train")
            #2° Grafico (Arancione)
            if self.development:
                pl.plot(epoch, self.accuracy_test, label="Test",linestyle='dashed')
            else:
                pl.plot(epoch, self.accuracy_validation, label="Validation",linestyle='dashed')
            pl.xlabel("Epoch")
            pl.ylabel("Accuracy")
            pl.title('Curva Apprendimento')
            #inserisce la griglia nel grafico
            pl.grid()
            #consente di salvare il plot in una directory
            pl.savefig(PATH + '/plots/Accuracy_Score'+output_file+'.png')
            #Il comando plt.legend consente di inserire una legenda nel grafico e determinarne la posizione
            #per i posizionamenti alternativi della legend e per il controllo delle sue ulteriori proprietà consultare help(pyplot.legend)
            pl.legend(loc="upper right") 
            
        
        

    
    """
    metodo per la valutazione di un test set: Prende in input il test_set_input, il test_set_output, e la funzione di costo
    e reestituisce in output l'errore sull'output layer e l'errore totale medio'
    """
    def evaluate(self, test_set_input, test_set_output, loss_function):
        layers,net = self.__forward(test_set_input)
        error, loss = self.loss(loss_function,test_set_output.T, layers[-1])
        mean_total_error = loss.sum()/ test_set_input.shape[0]
        return error, mean_total_error
    
    
    
    """    
    Metodo Grid-Search
    il metodo seguente prende in input i possibili iperparametri, li usa per addestrare dei modelli e ne calcola l'errore sul validation set.
    fornisce in output la lista di errori e iperparametri in ordine crescente rispetto al validation error.
    permette di compiede una grid search con i metodi 'batch','online','minibatch' e eventualmente di usare la cross-validation indicando il numero di fold
    permette anche di scegliere tra i metodi di inizializzazione 'xavier','random','he', 'typeone'
    Permette di scegliere  anche una o più funzioni di attivazione e una singola loss_function
    """
    def grid_search(self, methods, hidden_units, learning_rates, lambdas=[0], alphas=[0], epochs=2000, batch_size = 1, cross_val=False, k=1, init_methods = ["random"], act_functs = ["sigmoid"], loss_funct = "mse"):
        assert isinstance(methods,list) and isinstance(hidden_units,list) and isinstance(learning_rates,list) and isinstance(lambdas,list) and isinstance(alphas,list)  and isinstance(init_methods,list) and isinstance(act_functs,list), "The hyperparameters variables must be of 'list' type"
        self.grid_search_flag = True
        coarse_error=[]
        self.epochs = []
        length = len(methods)*len(hidden_units)*len(learning_rates)*len(lambdas)*len(alphas)*len(init_methods)*len(act_functs)
        i=0
        print ("Starting the cycle of grid-search")
        self.__print_progress_bar(i, length,prefix = 'Progress:', suffix = 'Complete')
        for init_method in init_methods:
            for act_funct in act_functs: 
                for method in methods:
                    for hidden_unit in hidden_units:
                        for learning_rate in learning_rates:
                             for lamb in lambdas:
                                 for alpha in alphas:
                                    self.weight_cleaning()
                                    parameters =[]
                                    parameters = [method,hidden_unit,learning_rate, lamb,alpha]
                                    self.weight_initialization(init_method,hidden_unit)
                                    if not cross_val:
                                        self.training(method,loss_funct,act_funct,learning_rate = learning_rate, lamb=lamb, alpha=alpha, epoch=epochs, batch_size = batch_size, early_stopping=True)
                                        mean_total_error = self.mean_total_error[-1]
                                        mean_validation_error = self.validation_error[-1]
                                        tupla = [mean_validation_error,mean_total_error,parameters,init_method,act_funct,self.epoch]
                                    else:
                                        assert k>1, "If you want to use the Cross-Validation you should specify a number of folds higher than 1"
                                        mean_validation_error, training_error = self.cross_validation(k,parameters,init_method, act_funct = act_funct, loss_funct = loss_funct, epoch=epochs, batch_size = batch_size)
                                        tupla = [mean_validation_error, training_error, parameters,init_method,act_funct,self.epoch]
                                    if mean_validation_error != "error":
                                        coarse_error.append(tupla)
                                    i += 1
                                    self.__print_progress_bar(i, length,prefix = 'Progress:', suffix = 'Complete')
                            
        print("Selected the best set of hidden-nodes and learning rates")
        coarse_error = sorted(coarse_error, key=lambda x: x[0])
        best_errors = coarse_error
        self.grid_search_flag = False

        return best_errors

              
    """
    Il metodo prediction effettua la predizione dato un input utilizzando l'ultimo modello addestrato.
    Il metodo fornisce la possibilità di normalizzare i dati di input utilizzando le statistiche usate per normalizzare il
    dataset originale. 
    In ultimo, se in fase di addestramento l'output era normalizzato, il presente metodo denormalizza l'output predetto
    usando le statistiche utilizzate per normalizzare l'output originale.
    L'ouput per un task di classificazione è specifico per ogni funzione di attivazione relativa al layer di output'
    """        
    def prediction(self,t_input, normalization = False):
        if normalization:             
            method = self.normalization_method
            if method == "simple":
                t_input = np.divide(t_input ,np.abs(self.original_input).max(axis=0))           
            if method == "minmax":
                t_input = np.divide( (np.subtract(t_input, (self.original_input).min(axis=0))) , np.subtract((self.original_input).max(axis=0), (self.original_input).min(axis=0)))
            if method == "z-score":
                t_input = (t_input - np.mean(self.original_input, axis=0)) / (np.std(self.original_input, axis=0))            

        layers,net = self.__forward(t_input)
        if self.task == "classification":
            if self.output_nodes == 1:
                if self.activation_function == "sigmoid":
                    output_layer = np.rint(layers[-1])
                if self.activation_function == "tanh":
                    output_layer = np.sign(layers[-1])
                if self.activation_function == "linear":
                    output_layer = layers[-1]

        elif self.task == "regression" and self.normalization_flag:
            output_layer = self.__denormalize(layers[-1])
        else:
            output_layer = layers[-1]
        return output_layer
    
    


    """
    metodo per il calcolo dell'accuracy dati l'output predetto e quello atteso
    """
    def __accuracy(self, predicted_output, test_output):
        if self.task == "classification":
            if self.output_nodes == 1:
                    predicted_output = predicted_output.T
                    #il metodo unique restituisce i type in ordine crescente:
                    difference = np.subtract(test_output, predicted_output)
                    correct = np.count_nonzero(difference==0)
                    #questo ci permette di lavorare sia con valori 0/1 che con -1/1 dove false(0,-1) e true(1,1)
                    accuracy = correct/len(difference)
                    return accuracy            



    """
    Dati gli iperparametri e il numero di fold il metodo effettua la cross-validation e restituisce l'errore relativo al validation e al training set'
    """     
    def cross_validation(self, k=5, hyp=["batch",[2,3],0.2,0,0], init_method = "xavier", act_funct = "sigmoid", loss_funct = "mse",epoch=100, batch_size = 1):
        self.error_cross_validation = []
        split_size = int(self.development_input.shape[0]/k)
        validation_error, training_error = 0 , 0
        for i in range(k):
            train_input = None 
            val_input = self.development_input[i*split_size:(i+1)*split_size]
            val_output = self.development_output[i*split_size:(i+1)*split_size]
            train_input = np.concatenate((self.development_input[:i*split_size],self.development_input[(i+1)*split_size:]))
            train_output = np.concatenate((self.development_output[:i*split_size],self.development_output[(i+1)*split_size:]))
            self.training_input = train_input
            self.training_output = train_output
            self.validation_input = val_input
            self.validation_output = val_output
            self.weight_cleaning()
            self.weight_initialization(init_method, hyp[1])
            self.training(hyp[0], loss_funct, act_funct, alpha=hyp[4], learning_rate= hyp[2], lamb=hyp[3], epoch=epoch , early_stopping=True, batch_size = batch_size)
            validation_error+= self.validation_error[-1]                                
            training_error += self.mean_total_error[-1]
            self.error_cross_validation.append(self.validation_error[-1])
        training_error /= k
        validation_error /= k
        return validation_error, training_error
        
    
    
   
    """
    Il seguente metodo serve solo per stampare la barra di caricameto
    """
    def __print_progress_bar (self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
