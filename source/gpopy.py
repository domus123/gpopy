#Generic Algorithm Hyper Parameter Optimization Python (GPOPY)
#writted by Lucas Guerra Borges
#lu.guerra7508@gmail.com
#-----------------------------------------------------------------# 
import random as rd
import pprint
import mlflow 
from types import FunctionType
from operator import itemgetter
from visualization import mlflow_tracking

__VERSION__ = 1.0.0
__GIT__ = "https://github.com/domus123/gpopy"

def header(): 
    """Print tunning header """
    print("GPOPY")
    print(f"VERSION: {__VERSION__}")
    print("HELP US IMPROVE")
    print(f"BUGS AND SUGESTION AT : {__GIT__}")
    print("Added support to MLFlow tracking\n")

header()

def activate_param (param, attribute): 
    """ Evaluate what value should be evaluated coming from PARAMS structure
        Inputs:
            param: Is one of the keys defined in your param dict
            attribute: Is the value associate with this param
        return: 
            upadate: The parameter evaluated according with your definition 
    """
    if (isinstance(attribute, list) or isinstance(attribute, tuple)) :
        length = len(attribute)
        select_elem = rd.randint(0, length - 1)
        update = { param :  attribute[select_elem]}

    elif (callable(attribute)):
        update = { param :  attribute()}
    elif (isinstance(attribute, dict)) :
        func = attribute['func']
        func_param = attribute['params'] 
        update = { param: func(*func_param)} 
    else: 
        update = { param : attribute}
    return update 

class Tunning(): 
    
    first_parent = {}
    second_parent = {}
    population = None
    score_function = None
    top_score = 0
    generation = 0
    score_function = None
    top_model = None
    genetic_tree = []

    def __init__(self, params,  population_size = 2, maximum_generation = 20, mutation_rate = 0.25):
        self.params = params
        self.population_size = population_size
        self.maximum_generation = maximum_generation
        self.mutation_rate = mutation_rate
        print(f"{pprint.pprint(params)}")
       
    def run(self, save_model = False):
        """Automaticly run the algorithm with maximum number of generation """
        for i in range(self.maximum_generation): 
            self.gen_population()
            self.score(save_model)
            print(f"##### Generation {i}   |   TopScore {self.top_score} #####")

    def create_individue(self) : 
        """ Create new individue based on self.params given when the class was created"""
        individue = {"score" : 0} 
        for param in self.params: 
            attribute = self.params[param]
            update = activate_param(param, attribute)
            individue.update(update)
        return individue
    
    def print_population(self):
        pp = pprint.PrettyPrinter(indent = self.population_size)
        pp.pprint(self.population)

    def gen_population(self):
        """Hold the creation of new individues"""
        
        if (self.first_parent == {}) and (self.second_parent == {}) :
            pop = []
            for i in range(self.population_size): 
               pop.append(self.create_individue())
            self.population = pop
            return self.population  #opt 

        else: 
            self.new_generation()
            return self.population
        
    def new_generation(self):
        """Create a new generation that already have top score individues """
        mutation1 = self.mutation(self.first_parent)
        mutation2 = self.mutation(self.second_parent)
        cross1, cross2 = self.crossover()
        self.population = [] 
        self.population.append(mutation1)
        self.population.append(mutation2)
        self.population.append(cross1)
        self.population.append(cross2)
        for i in range(self.population_size): 
            self.population.append(self.create_individue())
    
    def mutation(self, individue):
        """ Put mutation on individue """
        new_individue = {}
        for param in individue :
            mutate = rd.random()
            if (mutate > self.mutation_rate) and param != 'score':
                mutation = activate_param(param, self.params[param])
                new_individue.update(mutation)
            else :
                new_individue.update({param : individue[param]})
        return new_individue

    def crossover(self) :
        """ Crossover both parents"""
        numb_of_params = len(self.params) 
        half = numb_of_params / 2 
        first_crossover = {} 
        second_crossover = {} 
        
        for i, param in enumerate(self.params): 
            if i < half : 
                first_crossover[param] = self.second_parent[param] 
                second_crossover[param] = self.first_parent[param]
            else : 
                first_crossover[param] = self.first_parent[param]
                second_crossover[param] = self.second_parent[param]

        first_crossover['score'] = 0
        second_crossover['score'] = 0
        return first_crossover, second_crossover

    def set_score(self, score):
        """ change score function for individue """
        self.score_function = score
    
    def score(self, save_model = False):   
        """ Calculate the score returned from score_function and save score with/without model"""      
        if self.score_function == None: 
            assert False, "No score function setted, you can set it using set_score(func) or passing score= func during class instantiation"

        for i in self.population: 
            print(f"Scoring on ... {i}")
            if save_model:
                (score, model) = self.score_function(i)
                #TODO Change the > for any custom predicate 
                if (score > self.top_score) :
                    #save only the BEST model in memory, since keep then all may be memory expensive with big models
                    print("###################################################################")
                    print(f"New optimal model founded with a score of {score} ")
                    print("###################################################################")
                    self.top_model = (score, model)
                i['score'] = score
            else :
                i['score'] = self.score_function(i)
        self.generation +=1 
        sorted_list = sorted(self.population, key= itemgetter('score'), reverse= True)
        self.first_parent = sorted_list[0]
        self.second_parent = sorted_list[1]
        self.top_score = self.first_parent['score']
        self.genetic_tree.append(self.first_parent)

class FlowTunning(Tunning): 
    """ 
        This class was designed for using mlflow as a visualization tool and for saving your model
        
        Since you're using MLFlow, you should set in your environment the details such as artifacts, experiments and enviroment.
        You can set it before calling tunning. 

        What we wanted with this class is a way to keep track of informations and use it to visualize.
        You can track only the hyper parameters evolutions over the generations, or you can use a full MLFlow approach with your custom score and run functions.
    """
    def __init__(self, params, population_size=2, maximum_generation=20, mutation_rate=0.25, experiment_name="", auto_track=True):
        """ 
            Track_best: Boolean 
            The default value assumes that you don't customize your model with MLFlow and is using only GPOPY parameters as comparision,
            and for saving your models (In memory).
            More details on advanced logs and how mlflow can save your models MLFlow documentation at: https://www.mlflow.org/docs/latest/index.html
        """
        super().__init__(params, population_size, maximum_generation, mutation_rate)
        self.experiment_name = experiment_name
        self.tracking = auto_track 

    def mlflow_tracking(self, params, score, generation="None"):
        """
        Get the params of the network and log it to mlflow with a few parameters>
        Inputs: 
            params: A dict with all the values that was used in the algorithm 
            score: The result from running the model with 
        Return: None 
        """
        print(f"Params: {params}")
        #score, model = top_score 
        print(f"Score: {score}")
        params.pop('score', None) #Removing the model that came with params
        with mlflow.start_run():
            mlflow.set_tag("Generation", generation)
            mlflow.log_param("Generation", generation)
            mlflow.log_params(params)
            mlflow.log_metric("Score", score)

    def run(self): 
        print("Running while logging")
        for i in range(self.maximum_generation): 
            self.gen_population()
            if self.tracking: 
               self.easy_score(i)
            else: 
                self.detailed_score(i) 

    def easy_score(self, generation): 
        """
        Let GPOPY handle saves and scores 
        #TODO: Change this method, seems duplicated 
        """
        if self.score_function == None: 
            assert False, "No score function setted, you can set it using set_score(func) or passing score= func during class instantiation"
        top_params = {} 
        for i, elem in enumerate(self.population): 
            score, model = self.score_function(elem)
            if score > self.top_score: 
                print(f"*** New optimal model founded with a score of {score} ***")
                self.top_score = score
                self.top_model = (score, model)
            elem['score'] = score

        sorted_list = sorted(self.population, key= itemgetter('score'), reverse= True)
        self.first_parent = sorted_list[0]
        self.second_parent = sorted_list[1]
        generation_top_score = self.first_parent['score']
        self.genetic_tree.append(self.first_parent)
        print(f"Better parent {self.first_parent}")
        print("DONE")
        self.mlflow_tracking(self.first_parent, generation_top_score, generation + 1) ## Simple version using mlflow 

    def detailed_score(self, generation): 
        """
        A more detailed run with mlflow.
        While use this function, is recomended that you use the mlflow tracking function for the model you're using.
        e.g You can use mlflow tracking on Tensorflow, torch, scikit ... 
        
        TODO: Merge this function with easy_score, and pass the paramater used on run as parameter for easy_score
        This fix will be available 
        """
        if self.score_function == None: 
            assert False, "No score function setted, you can set it using set_score(func) or passing score= func during class instantiation"
        for i, elem in enumerate(self.population): 
            with mlflow.start_run() as run: 
                tags = {
                    'generation': generation + 1, 
                    'individue' : i  + 1
                }
                mlflow.set_tags(tags)
                mlflow.log_param("Generation", generation + 1)
                score, model = self.score_function(elem)
            if score > self.top_score: 
                print(f"*** New optimal model founded with a score of {score} ***")
                self.top_score = score
                self.top_model = (score, model)
            elem['score'] = score
        sorted_list = sorted(self.population, key= itemgetter('score'), reverse= True)
        self.first_parent = sorted_list[0]
        self.second_parent = sorted_list[1]
        generation_top_score = self.first_parent['score']
        self.genetic_tree.append(self.first_parent)
        print(f"Better parent {self.first_parent}")
        print("DONE")
        
