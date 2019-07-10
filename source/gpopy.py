#Generic Algorithm Hyper Parameter Optimization (GHPO)
#writted by Lucas Guerra Borges
#lu.guerra7508@gmail.com
#-----------------------------------------------------------------# 
import random as rd
import pprint
from types import FunctionType
from operator import itemgetter

__VERSION__ = 0.1
__GIT__ = "https://github.com/domus123/gpopy"

def tunning_header(): 
    """Print tunning header """
    print("GPOPY")
    print(f"VERSION: {__VERSION__}")
    print("HELP US IMPROVE")
    print(f"BUGS AND SUGESTION AT : {__GIT__}")

def activate_param (param, attribute): 
    """ evaluate what value should be evaluated coming from PARAMS structure"""
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
    score_function = None
    top_model = None
    genetic_tree = []

    def __init__(self, params,  population_size = 2, maximum_generation = 20,mutation_rate = 0.25 ):
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
        if self.score_function == None : 
            assert False, "No score function setted, you can set it using set_score(func) or passing score= func during class instantiation"

        for i in self.population: 
            print(f"Scoring on ... {i}")
            if save_model:
                (score, model) = self.score_function(i)
                if (score > self.top_score) :
                    #save only the BEST model in memory, since keep then all may be memory expensive with big models
                    print("###################################################################")
                    print(f"New optimal model founded with a score of {score} ")
                    print("###################################################################")
                    self.top_model = (score, model)
                i['score'] = score
            else :
                i['score'] = self.score_function(i)

        sorted_list = sorted(self.population, key= itemgetter('score'), reverse= True)
        self.first_parent = sorted_list[0]
        self.second_parent = sorted_list[1]
        self.top_score = self.first_parent['score']
        self.genetic_tree.append(self.first_parent)


tunning_header() 