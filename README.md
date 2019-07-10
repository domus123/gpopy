<h1> GPOPY </h1>
<h2> GPOPY is a way to optimize parameters in a easy way </h2> 

<p> GPOPY was build using a genetic algorithm, that will handle parameter optimizations for you in any problem </p> 

<h5> Declaring parameters </h5>
<h6> In order to GPOPY understand your parameters, they must be declared inside of a dict file </h6>


```python 

#taken from model.py example
PARAMS = {                                                                                                                                                                             
    'batch_size' : [ 8,16,32,64,128,256],                                                                                                                                           
    'epochs' : [8, 16, 32, 64],                                                                                                                                                        
    'dense_layers' : [64, 128, 256, 512],                                                                                                                                              
    'dropout' : {                                                                                                                                                                      
        'func' : random.uniform,                                                                                                                                                       
        'params' : [0.3, 0.7]                                                                                                                                                          
    },                                                                                                                                                                                 
    'activation' : 'relu',                                                                                                                                                             
    'learning_rate': {                                                                                                                                                                 
       'func' :  random.uniform,                                                                                                                                                       
       'params' : [0.01, 0.0001]                                                                                                                                                       
    },                                                                                                                                                                                 
    'filters' : [10, 64, 128],                                                                                                                                                         
    'use_bias' : [True, False]                                                                                                                                                         
}     

```
<h6> List parameters and tuple parameters </h6>

```python 
params = {
   'list_parameter' : [1,2,3,4,5] ,
   'tupple' : (1,2) 
   } 
   #may return any value in the list and tupple
```

When running a list and/or tuple parameter, GPOPY will select from one of the values from then.

<h6> Function parameter </h6>

``` 
   PARAMS = {
    'random_function' : random.random,     
    }
    #gpopy will evaluate random.random at each iteration
```

When passing a function without arguments, GPOPY will evaluate the function in each iteration, and use the result as parameter to pass to your function.

<h6> Dict function objects </h6>

```python
#random.uniform(0.01, 0.02)
    PARAMS = {
        'dict_func_object' : {
            'func' : random.uniform, 
            'params' : [0.01, 0.02] 
        } 
    }
```

The only dict we currently accept is with a function with a multiple parameters.
This was created to make it easier to call functions that need parameters, as for example, random.uniform.
The parser inside the code need the params to be 'func' and 'params.

<h6> Other types of values </h6>
If you use other type of values, for example int, double, string, they will be returned.
This type is good when you already now a parameter that you don't want to change.

<h5> Using GPOPY </h5>

<h6> Constructor </h6>

```python
Tunning(params, population_size = 4, maximum_generation = 20, mutation_rate = 0.25)

```
<p> params: </h5>  Dict with all the params that will be used to generate your population
<p> score: Function that will use as parameter the individue and return an score (e.g Keras model)
<p> population_size: Number of unique individues generated at each generation (not included crossover and mutation individue)
<p> maximum_generation: number of generations that will be run (in case you use run method)
<p> mutation_rate : rate at which mutation occur, lower mutation_rate means more mutations 

<h5> Setting up GPOPY </h5>

```python

#Automaticly running 
tunning = Tunning(PARAMS, population_size = 2) 
tunning.set_score(model) 
results = tunning.run()  #will run model with a generation_number of 20

```

<h5> Running with more liberty </h5> 

```python

tunning = Tunning(PARAMS, population_size = 2)
tunning.set_score(model)
needed_accuracy = 0.95

for i in range(10): 
    """Will run 10 times or until find the needed accuracy """
    tunning.gen_population() #create new individue/mutation/crossovers
    tunning.score() #run score functions and save the results from them
    if tunning.top_score >= needed_accuracy: 
        break 
top_score = tunning.top_score
best_match = tunning.first_parent 
second_best_match = tunning.second_parent
```

<h6> GPOPY has in-line documentation, few free to read and change it when needed, all tell us about your changes so we can keep getting better!</h6>

<h5>Creating a score function </h5>
<p> In order to be a score functions, we just need it to receive our data as parameter and return a score.
<p> In this implementation of the algorithm, we will get the highest score for evolving.
<p> Later we can change it

```python
    PARAMS = {
        'a' : [1,2,4,8,16,32] ,
        'b' : {
            'func' : random.rantint,
            'params' : (0, 100)
        'x' : random.random
        }
        
     def model(data): 
        #unwrapping 
        a = data['a']
        b = data['b']
        x = data['x']
        return a * (x ** 2) + b * x
    }
```

<h5>Saving your ML/DL models</h5>
<p> Since most of ML and DL are instantiated with random weights, some times under the same circunstance (PARAMS) we can obtain different test results </p>
<p> For resolving this, we added a option to "score" function to save models and reuse it later </p>

```python

    tunning.score(save_model= True)
    #or just
    #tunning.score(True)
    
    score, model = tunning.top_model #HA
    model.evaluate(x_test, y_test, verbose=0)

```

<h5> ATTENTION </h5>
<p> IF save_model= True, your score function should be returning a tupple, that constains (score, model), been model any function or value that you want to hold on memory </p>

```python 
    #model function returning from Keras 
    return (score[1], model)
```


As was told earlier, GPOPY run Genetic algorithm on ANY function that meet those especifications, not only for ML and Neural Net Models.


<h4> Final Notes </h4>
<p> Few free to use, edit and make all the changes needed in the code </p>
<p> If you use it in your work, please give us a credit </p>
<p> If you have any changes and/or bug to report, please few free to talk with us </p>
<p> You can personally contact me at lu.guerra7508@gmail.com feel free to be in touch </p>
<p> GPOPY is part of one paper i'm writing, so much more optimization and changes are coming </p>
<p> THANK YOU </p>


