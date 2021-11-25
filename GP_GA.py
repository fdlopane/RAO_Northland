# RESOURCE ALLOCATION OPTIMISATION
# Author: Fulvio D. Lopane

# Started writing: 11/07/2017

program_name = "GP_GA"

""" IMPORT MODULES
"""

import os
os.system('cls')  # clears screen

import time
import os.path

start_time = time.asctime()
running_time_start = time.clock()

print "Program: GP_GA"
print "Starts at: " , start_time
print
print "Importing modules..."

# DEAP modules to facilitate the genetic algorithm
from deap import algorithms
from deap import base 
from deap import creator # creates the initial individuals
from deap import tools # defines operators

# Modules to handle rasters into arrays
import rasterIO

# Module to handle arrays
import numpy as np

# Module to handle math operators
import math
import random as rndm
from copy import copy

# Module to handle .csv files
import pandas as pd

# Module to handle shape files
import fiona

# Modules for the spatial optimisation framework

import Initialise_01 as Init # initialisation module
import Evaluate_01 as Eval # Module to calculate and return fitnesses
import Constraints as Constraint
import Outputs_02 as Output

import gc
gc.disable()

Modules = ['Initialisation', Init.__name__, 'Evaluation', Eval.__name__, 
          'Constraints', Constraint.__name__, 'Output', Output.__name__]
		   
print "All modules imported."
print

""" DIRECTORIES
"""
# on P: drive:
Data_Folder 	= "P:/RLO/Python_Codes/GP_NZ_Data/Northland/"
Results_Folder	= "P:/RLO/Python_Codes/NZ_Case_Study/Results/"
External_Results_Folder = "P:/RLO/Python_Codes/NZ_Case_Study/Results/"

""" PROBLEM FORMULATION - General Parameters
"""

region_name = 'Northland'
Tot_Population = 151638.0	# Total population living in the case study area

# Variables for the search
Clinics_Max = 20		# Maximum number of clinics
Clinics_Min = 10		# Minimum number of clinics

# Initialisation parameters
Ratio_Doc_Pop = 1.0/1500  # Target ratio Doctor/patients
Min_Doc_factor = 0.5	# Multiplication factor of "optimal n of doctors" to determine minimum number of doctors that can be assigned to a clinic
Max_Doc_factor = 1.5	# Multiplication factor of "optimal n of doctors" to determine maximum number of doctors that can be assigned to a clinic

# Evaluation parameters
Wc = 4	# Weight factor for number of clinics. Value of 4 means that 1 clinic = iether 6000 or 100000 patients (according to target ratio) 
Wd = 1	# Weight factor for number of doctors

X_quantile = 0.9 # Quantile for travel time opt function. (i.e. how much time XX % of patients reach the closest clinic)

# When fdist = (W_A * TT_Average) + (W_T * TT(>Threshold)_Average)
TT_Threshold = 10 # Travel time threshold below which I consider "close enough" in the evaluation phase
W_A = 1 # Weigth factor for Travel Time average
W_T = 1 # Weigth factor for Travel Time(>threshold) average

GEUD_power = 2

# Constraints paramteres
Min_doc = int(0.75*(Tot_Population*Ratio_Doc_Pop))	# Minimum number of doctors that must be present in a solution = HALF than optimal
Max_doc = int(1.25*(Tot_Population*Ratio_Doc_Pop))	# Maximum number of doctors that must be present in a solution = 1.5 times than optimal

Problem_Parameters = 	['Maximum clinics n', Clinics_Max, 'Minimum clinics n', Clinics_Min, 'Optimal ratio doctor/patients', Ratio_Doc_Pop,
						'Weight factor for number of clinics', Wc, 'Weight factor for number of doctors', Wd, 'Quantile for TravTime opt function', X_quantile,
						'Weight factor for travel time average', W_A, 'Weight factor for travel time (>threshold) average', W_T, 'Travel time treshold (min)', TT_Threshold,						
						'Min n of doctors in a potential solution', Min_doc, 'Max n of doctors in a potential solution', Max_doc]


#LOOKUP
# To handle the constraints the algorithm uses a lookup for proposed allocation sites.
# The lookup list contains the locations of sites with an existing clinic or available for a new one.

if os.path.isfile(os.path.join(Results_Folder, "lookup.txt")):
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
	print "Lookup uploaded."
else:
	File_centroids = "Available_centroids.shp"
	Lookup = Init.Generate_Lookup(Data_Folder, Results_Folder, File_centroids)

# So we know how long to make the chromosome
No_Available = len(Lookup) # number of sites with space for development
print "Number of available cells: " , No_Available 


# OUTPUT VARIABLES
# For results 
Sols, Gens = [],[] # Saves all solutions found, saves each generation created                       
# Keep a record of the retaintion after constraints
start = [] # initial array
# Resave the files to contain the arrays
np.savetxt(Results_Folder+'N_doctors_Constraint.txt', start,  delimiter=',', newline='\n')


""" TYPES - creating fitness class, negative weight implies minimisation 
"""

# FITNESS - Defining the number of fitness 
# objectives to minimise or maximise
# Creating types (Fitness, Individual), DEAP documentation: http://deap.readthedocs.io/en/master/tutorials/basic/part1.html
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) # -1.0 for each objective to minimise

# INDIVIDUAL
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMin) #typecode b = integer


""" INITIALISATION - Initially populating the types
"""

toolbox = base.Toolbox()

def Generate_C_Plan(Ind, Clinics_Max, Clinics_Min):
	# This function takes as an argument "Ind" which is the creator.individual that takes as an argument the clinics plan
	# and returns the Individual in which is saved a potential solution
	# Clinics_Plan is a list of 0s and 1s, where 1=assigned clinic. To know the coordinates make reference to Lookup list.
	
	Clinics_Plan = Init.Generate_ClinicsPlan(No_Available, Clinics_Max, Clinics_Min, Results_Folder, Data_Folder, Ratio_Doc_Pop, Min_Doc_factor, Max_Doc_factor)

	return Ind(Clinics_Plan) 

toolbox.register("individual", Generate_C_Plan, creator.Individual, Clinics_Max, Clinics_Min) # creates an individual = single potential solution
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # creates a population of individuals = set of potential solutions

# Save two variables containing the min and the max possible doctors for each available cell
# I'll use these for the mutation operator decorating function
Min_doctors_Plan, Max_doctors_Plan = Init.Generate_Min_Max_Doc(No_Available, Results_Folder, Data_Folder, Ratio_Doc_Pop, Min_Doc_factor, Max_Doc_factor)


"""  FUNCTIONS - Evaluate functions and constraint handling
"""

def Evaluate(Clinics_Plan):
	# Generate the evaluation of functions to minimise/maximise

	Proposed_Sites = Init.Generate_Proposed_Sites(Clinics_Plan, Results_Folder) # List of coordinates of proposed sites

	# Dist_Fit = Eval.Calc_fdist(Results_Folder, Proposed_Sites, X_quantile)
	# Dist_Fit = Eval.Calc_fdist_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan)
	# Dist_Fit = Eval.Calc_fdist_average_dist_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan)
	# Dist_Fit = Eval.Calc_fdist_threshold_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan, TT_Threshold, W_A, W_T)
	# Dist_Fit = Eval.Calc_fdist_GEUD_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan, GEUD_power)
	Dist_Fit = Eval.Calc_fdist_weight_pop(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan)
	
	# Cost_Fit = Eval.Calc_fcost(Clinics_Plan, Wc, Wd)
	Cost_Fit = Eval.Calc_fcost_constraint(Clinics_Plan, Wc, Wd, Min_doc, Max_doc) # Function with additional constraint
	# Cost_Fit = Eval.Calc_fcost_Diff_Opt_constraint(Clinics_Plan, Ratio_Doc_Pop, Tot_Population, Min_doc, Max_doc, Results_Folder, Data_Folder, Ratio_Doc_Pop)
	
	return Dist_Fit, Cost_Fit

Fitnesses = ['fdist', 'fcost']


def Track_Offspring():
    # Decorator function to save the solutions within the generators
    def decCheckBounds(func):
        def wrapCheckBounds(*args, **kargs):
            offsprings = func(*args, **kargs)
            # Append this generations offspring
            Gens.append(offsprings)
            for child in offsprings:
                # attach each individual solution to solution list. Allows the
				# demonstration of which solutions the Algorithm has investigated.
                Sols.append(child)
            return offsprings
        return wrapCheckBounds
    return decCheckBounds  


""" OPERATORS - Registers Operators and Constraint handlers for the GA
"""

## Evaluator
# Evaluation module - so takes the development plan
toolbox.register("evaluate", Evaluate)

## EVOLUTIONARY OPERATORS
# Designate the method of crossover
# essentialy takes two points along the array and swaps the clinics
# Between them. Designating the string name for the output text document
Crossover = "tools.cxTwoPoint"
#toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mate", tools.cxTwoPoint)
# Designate the method of mutation
# Decided to use mutShuffleIndexes which merely moves the elements of the array around
Mutation = "tools.mutShuffleIndexes, indpb=0.1" # indpb - Independent probability for each attribute to be exchanged to another position.
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)

# Selection operator. Either use this or SPEA2 as dealing with multiple OBjs
# See DeB, 2002 for details on NSGA2
Selection = "tools.selNSGA2"
toolbox.register("select", tools.selNSGA2)

Operators = ['Selection', Selection, 'Crossover', Crossover, 'Mutation', Mutation]

# CONSTRAINT HANDLING
# Using a decorator function in order to enforce a constraint of the operation.
# This handles the constraint on the total numBer of clinics. So the module
# interrupts the selection phase and investigates the solutions selected. If 
# they fail to exceed the minimum clinics number or exceed the max clinics number
# its deleted from the gene pool.   
# Moreover to this, each generation is saved to the Gen_list and each generated
# Solution is saved to a sol_list. This for display purposes.


# Constraint to ensure the number of clinics falls within the targets
toolbox.decorate("select", Constraint.Check_N_Doctors(Min_doc, Max_doc, Results_Folder), Track_Offspring())

toolbox.decorate("mate", Constraint.Mutation_constraint(Min_doctors_Plan, Max_doctors_Plan))
toolbox.decorate("mutate", Constraint.Mutation_constraint(Min_doctors_Plan, Max_doctors_Plan))

# toolbox.decorate("select", Track_Offspring())

## OPTIMISATION PARAMETERS
MU      = 1000	# Number of individuals to select for the next generation
NGEN    = 50    # Number of generations
LAMBDA  = 1000  # Number of children to produce at each generation
CXPB    = 0.6   # Probability of mating two individuals
MUTPB   = 0.3   # Probability of mutating an individual

GA_Parameters = ['Generations', NGEN, 'No of individuals to select', MU, 
                 'No of children to produce', LAMBDA, 'Crossover Probability',
                 CXPB, 'Mutation Probability', MUTPB]


def Genetic_Algorithm():    
    # Genetic Algorithm    
    print "Beginning GA operation"
    
    # Create initialised population
    print "Initialising"
    pop = toolbox.population(n=MU)
    
    # hof records a pareto front during the genetic algorithm
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", tools.mean)
    #stats.register("std", tools.std)
    stats.register("min", min)
    #stats.register("max", max)
    
    # Genetic algorithm with inputs
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats= stats, halloffame=hof)
                                                     
    return hof  


if __name__ == "__main__":
	# Returns the saved PO solution stored during the GA
	hof = Genetic_Algorithm()
	
	
	Complete_Solutions = copy(Sols)
	for PO in hof:
		Complete_Solutions.append(PO)
	
	# Update the results folder to the new directory specifically for this run
	Results_Folder = Output.New_Results_Folder(Results_Folder)    
	
	# Format the solutions so they are compatible with the output functions
	# Gives each a number as well as added the fitness values to from:
	# [ Sol_Num, Sites, Fitnesses]
	frmt_Complete_Solutions = Output.Format_Solutions(Complete_Solutions)
	
	# Extract the minimum and maximum performances for each objective
	# To allow for solutions to be normalised
	MinMax_list = Output.Normalise_MinMax(frmt_Complete_Solutions)
	
	# Normalise the formatted Solution list using the Min and Maxs for 
	# each objective function    
	Normalised_Solutions = Output.Normalise_Solutions(MinMax_list, frmt_Complete_Solutions)
	
	# Extract all the Pareto fronts using the normalised solutions
	Output.Extract_ParetoFront_and_Plot(Normalised_Solutions, True, External_Results_Folder, Results_Folder, Data_Folder)
	
	# Extract all the Pareto fronts using the solutions retaining their true values.
	Output.Extract_ParetoFront_and_Plot(frmt_Complete_Solutions, False, External_Results_Folder, Results_Folder, Data_Folder)
	
	# Output a file detailing all the run parameters
	running_time_end_s = time.clock() # running time in seconds
	running_time_end_minutes = running_time_end_s/60 # running time in minutes
	run_time = str(int(running_time_end_minutes))
	Output.Output_Run_Details(External_Results_Folder, Results_Folder, Modules, Operators, Problem_Parameters, GA_Parameters, Fitnesses, run_time, region_name)

	# Create Shapefiles
	# Output.Create_sol_shapefile(Results_Folder, External_Results_Folder)
	
	# Create Distance .csv files for statistics
	Output.Save_dist_Clinics_Meshblocks(External_Results_Folder, Results_Folder)
	
	# GENERATIONS OUTPUTS
	
	# Create a new array to hold the formatted generations
	frmt_Gens = []    
	for Gen in Gens:
		# For each generation, format it and append it to the frmt_Gens list
		frmt_Gens.append(Output.Format_Solutions(Gen))
	
	Output.Extract_Generation_Pareto_Fronts(frmt_Gens,MinMax_list, Results_Folder, Data_Folder, External_Results_Folder)
	
	# Generate tables containing the single solutions of the Pareto front
	Output.Generate_Solutions_Tables(Results_Folder, External_Results_Folder)
	
	
	end_time = time.asctime()
	
	print "END. end time = ", end_time
	print "Running time = ", int(running_time_end_minutes), " minutes"
	quit()