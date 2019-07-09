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
    PARAMS = {
        'dict_func_object' : {
            'func' : random.uniform, 
            'params' : [0.01, 0.02] 
        } 
    }
    
#random.uniform(0.01, 0.02)
```
The only dict we currently accept is with a function with a multiple parameters.
This was created to make it easier to call functions that need parameters, as for example, random.uniform.
The parser inside the code need the params to be 'func' and 'params.

<h6> Other types of values </h6>
If you use other type of values, for example int, double, string, they will be returned.
This type is good when you already now a parameter that you don't want to change.





