# -*- coding: utf-8 -*-
"""
Outputs

Module to handle the outputs of the Genetic Algorithm search
"""

import NonDom_Sort
import Initialise_01 as Init

import numpy as np
from copy import copy
from itertools import combinations
import os 
import matplotlib
import matplotlib.pyplot as plt #Allow the solutions to be plotted
# Facilitate the creation of a datetime stamp for the results
import datetime, time 
import rasterIO
import sys

import pandas as pd
from ast import literal_eval 
import geopandas as gpd
from shapely.geometry import Point 


# Creating a data time stamp
start_time = time.time()
date_time_stamp = datetime.datetime.fromtimestamp(start_time).strftime('%d%m%y-%H%M')

def Get_Label(Obj):
	#returns the label for the axis
	if Obj   == 0:  return r'$f_{dist} $'
	elif Obj == 1:  return r'$f_{cost} $'

	
def Get_String(Obj):
	if Obj   == 0:  return 'fdist'
	elif Obj == 1:  return 'fcost'

	
def Return_Per_Retention(Results_Folder):
	Clinics_Constraint_list   = np.loadtxt(Results_Folder+"N_doctors_Constraint.txt",delimiter=",")
	
	Av_Clinics_Retention  = np.sum(Clinics_Constraint_list)/ len(Clinics_Constraint_list)
		
	return Av_Clinics_Retention

	
def Output_Run_Details(External_Results_Folder, Results_Folder, Modules, Operators, Prob_Form, GA_Parameters,Fitnesses,run_time, region_name):
	# Function to output a text file detailing the parameters for the GA Search
	# Including:
		# + Data time stamp
		# + Modules used (so can see which versions were used)
		# + How the problem is formulated
		# + The objectives optimised 
		# + Operators used for the evolutionary operating

	OutFile = open(Results_Folder+'Output_file.txt', 'w')
	OutFile.write('Genetic Algorithm Run over '+region_name+' Case Study\n')
	OutFile.write('______________________________________________\n')
	
	OutFile.write('Run completed on '+date_time_stamp+'\n')  
	OutFile.write('Running time (minutes) = '+run_time+'\n')  
	OutFile.write('______________________________________________\n') 
	
	# Write the modules used
	OutFile.write('Modules Utilised: \n')
	for x in range(0, len(Modules), 2):
		OutFile.write('- '+Modules[x]+': '+Modules[x+1]+'\n')
	OutFile.write('______________________________________________\n') 
	
	# Write the parameters for the problem being solved
	OutFile.write('Problem Formulation:\n')
	for x in range(0,len(Prob_Form),2):
		OutFile.write('- '+Prob_Form[x]+': '+str(Prob_Form[x+1])+'\n')    
	OutFile.write('______________________________________________\n') 
	
	# Write the objective functions optimised
	OutFile.write('Objectives Optimised:\n')
	for x in range(0, len(Fitnesses)):
		OutFile.write(str(x+1) + ') ' + Fitnesses[x]+' \n')
	OutFile.write('______________________________________________\n') 
	
	# Write the operators used
	OutFile.write('Operators Utilised:\n')    
	for x in range(0, len(Operators),2):
		OutFile.write('- '+Operators[x]+': '+Operators[x+1]+'\n')
	OutFile.write('______________________________________________\n') 
	
	# Writing the retention rate of the constraints
	Av_Dwell_Retention = Return_Per_Retention(External_Results_Folder)
	OutFile.write('Average retention rate after Clinics Total Restraint is '+str(Av_Dwell_Retention)+'%\n')
	OutFile.write('______________________________________________\n') 
	
	# Write the search parameters for the Genetic Algorithm
	OutFile.write('GA Search Parameters:\n')    
	for x in range(0, len(GA_Parameters),2):
		OutFile.write('- '+GA_Parameters[x]+': '+str(GA_Parameters[x+1])+'\n')

	OutFile.close()
	
	print "Written Run Details File"

			
def Normalise_MinMax(Sol_List):
	# Function calculates the maximum and minimum value for each     
	MinMax_list = []
	
	# Calculate the number of objectives to calculate min and maxs for
	# by calculating the length of the first solutions fitness column
	No_Obj = len(Sol_List[0][2]) # Sol format = [Sol_num, Sol, fitnesses], so number of objectives = len(fitnesses)
	
	# For each Obj calculate their minimum and maximum and add them to the MinMax_list
	for ObjFunc in range(0, No_Obj):

		# Sort the solution list ascending by the objective         
		Sol_List.sort(key=lambda x: x[2][ObjFunc], reverse = False) #currently sorted by smallest 1st obj    
		# Extract the minimum solution        
		Obj_Min = Sol_List[0][2][ObjFunc]    
		Sol_List.sort(key=lambda x: x[2][ObjFunc], reverse = True) #currently sorted by largest 1st obj    
		Obj_Max = Sol_List[0][2][ObjFunc]     
		MinMax = (Obj_Min, Obj_Max)
		MinMax_list.append(MinMax)
		
	return MinMax_list
   
   
def Format_Solutions(Solutions):
	# Function formats the GA outputs into a form which can  be normalised and plotted.
	
	# Array to hold the new formatted solutions
	Frmt_Sols = []
	
	# Solution number count
	Sol_num = 0    
	
	for Sol in Solutions:
		fitnesses = []
		for Obj_Function in Sol.fitness.values:
			fitnesses.append(Obj_Function)
		frmt_sol = [Sol_num, Sol, fitnesses]
		Frmt_Sols.append(frmt_sol)
		Sol_num +=1
	return Frmt_Sols

	
def Normalise_Solutions(MinMax_list, Solutions):
	
	Copied_Solutions = copy(Solutions)
	Norm_List = []
	
	for solution in Copied_Solutions:
		new_solution = []
		new_solution.append(solution[0]) # Solution format = [Sol_num, Sol, fitnesses]
		new_solution.append(solution[1])
		Norm_Fitnesses = []
		for a in range(0, len(solution[2])):
			# Normalise the fitness using the Minimum and Maximum from the MinMax_list
			Min, Max = MinMax_list[a][0], MinMax_list[a][1]
			Norm_Obj = Norm(solution[2][a], Min, Max)
			Norm_Fitnesses.append(Norm_Obj)

		new_solution.append(Norm_Fitnesses)

		Norm_List.append(new_solution)
	
	return Norm_List
	
	
def Norm(value, Obj_Min, Obj_Max):
	Norm_value = (value - Obj_Min)/(Obj_Max - Obj_Min)
	return Norm_value

	
def frmt_CSV(Set):
	Copied_Set = copy(Set)
	New_Set= []
	for solution in Copied_Set:
		new_solution = []
		new_solution.append(solution[0])
		ObjFunc = np.array(solution[2])
		for x in ObjFunc:
			new_solution.append(x)
		New_Set.append(new_solution)
	New_Set=np.array(New_Set)
	return New_Set
 
 
def Save_to_CSV(Set, Obj_list, Norm, Results_Folder):
	Output_File = 'PO_Set_'
	if Norm == True:
		Output_File += 'Norm_'
	for Obj in Obj_list:
		Obj_String  = str(Get_String(Obj))
		Output_File = Output_File+Obj_String+'_'
	frmt_Set = frmt_CSV(Set)
	File = str(Results_Folder)+str(Output_File)+"_Fitness.csv"
	
	np.savetxt(File, frmt_Set,  delimiter=',', newline='\n')
	
	
def Save_Pareto_Set(PO_Set, Obj_list, Norm, Results_Folder):
	Output_File = 'PO_Set_'
	if Norm == True:
		Output_File += 'Norm_'
	for Obj in Obj_list:
		Obj_String  = str(Get_String(Obj))
		Output_File = Output_File+Obj_String+'_'
	
	f = open(Results_Folder+Output_File+".txt",'w')
	for sol in PO_Set:
		f.write(str(sol)+'\n')
	f.close()

	
def Plot_Format(Set):
	Copied_Set = copy(Set)
	New_Set= []
	for solution in Copied_Set:
		new_solution = []
		ObjFunc = np.array(solution[2])
		for x in ObjFunc:
			new_solution.append(x)
		New_Set.append(new_solution)
	New_Set=np.array(New_Set)
	return New_Set    

	
def Plot(Pareto_Set, MOPO, Solutions, X_Axis, Y_Axis, Results_Folder, Norm):
	# Format it to a numpy array type so [:, Axis] works
	# Returns in the form [SolNum Obj1, Obj2, ...]
	Pareto_Set  = Plot_Format(Pareto_Set)
	MOPO        = Plot_Format(MOPO)
	Solutions   = Plot_Format(Solutions)
	
	# Plot Solutions, Pareto set, multi obj Pareto set and current development trend
	plt.plot(Solutions[:,X_Axis],Solutions[:,Y_Axis], "^", color = "blue", markersize=2, label = "Solutions")  #old markersize = 5
	 
	plt.plot(Pareto_Set[:,X_Axis],Pareto_Set[:,Y_Axis], "--", color = "red", linewidth=1 , label = "Pareto front")    # old linewidth=2
	
	X_Axis_label = Get_Label(X_Axis)
	Y_Axis_label = Get_Label(Y_Axis)
	
	# plt.title("Non Dominated front of a set solutions")
	plt.xlabel(X_Axis_label, fontsize = 18)
	plt.ylabel(Y_Axis_label, fontsize = 18)
	
	# fig = plt.figure()
	
	if Norm == True:
		# plt.xlim(xmin=0)  # adjust the min leaving max unchanged
		# plt.ylim(ymin=0)  # adjust the min leaving max unchanged
		pass
	else:
		plt.xlim(xmin=0)  # adjust the min leaving max unchanged
		plt.ylim(ymin=0)  # adjust the min leaving max unchanged
		
		# Transform the X-axis unit from seconds to minutes
		# scale_x = 60
		# ticks_x = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
		# matplotlib.axis.XAxis.set_major_formatter(ticks_x)
		
		# pass
	
	plt.legend(loc=1)
	X_Axis, Y_Axis = Get_String(X_Axis), Get_String(Y_Axis)
	
	if Norm == True:
		plt.title("Normalised plot of Pareto-optimal solutions")        
	else:
		plt.title("Plot of Pareto-optimal solutions")
	
   
	if Norm == True:
		Output_File = Results_Folder+'Norm_Plot_'+X_Axis+'_against_'+Y_Axis+'_'+date_time_stamp+'.jpeg'        
	else:
		Output_File = Results_Folder+'Plot_'+X_Axis+'_against_'+Y_Axis+'_'+date_time_stamp+'.jpeg'
		
	plt.tight_layout() # automatically adjusts plot size
	
	plt.savefig(Output_File)
	# print "Plotted Objective ", X_Axis, ' against ', Y_Axis
	plt.clf()    
	plt.close    
	

def New_Results_Folder(Results_Folder):
	# Function to create a new folder within the results folder to handle the results of the specific run
	
	# Location of the new folder with 'Run' and the data time stamp attached
	New_Results_Folder = Results_Folder+'Run_'+date_time_stamp+'/'
	  
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_Results_Folder): os.makedirs(New_Results_Folder)  
	
	return New_Results_Folder

""" WRITE RASTERS	
def Write_Rasters(Set, Results_Folder, Data_Folder, External_Results_Folder):
	# Upload the details in order to save our output files  
	file_pointer        = rasterIO.opengdalraster(Data_Folder+'Empty_plan.tif')       
	driver, XSize, YSize, proj_wkt, geo_t_params = rasterIO.readrastermeta(file_pointer)
	
	# Specifying the format as EPSG:19971 New Zealand Transverse Mercator 2000
	epsg = 19971
	# print len(Set)
	
	# Lookup_local = Init.Generate_Lookup_local(Data_Folder, External_Results_Folder)
	
	for Solution in Set:
		# Extract the solution number
		Sol_Num = Solution[0]
				
		W_plan = Init.Generate_DevPlan(Solution[1], Data_Folder, External_Results_Folder) # Solution[1] contains the proposed sites
		
		W_Plan_Outfile = Results_Folder+'W_Plan'+str(Sol_Num)+'.tif'

		rasterIO.writerasterbands(W_Plan_Outfile, 'GTiff', XSize, YSize, geo_t_params, epsg, None, W_plan)
"""

""" CREATE SHAPEFILE
def Create_sol_shapefile(Results_Folder, External_Results_Folder):
	
	# Specifying the format as EPSG:19971 New Zealand Transverse Mercator 2000
	
	l = []
	with open(Results_Folder + "PO_Set_fdist_fcost_.txt") as f: # Result folder should already be updated to the Run folder
		for line in f.readlines():
			l.append(literal_eval(line))
	
	df0 =  pd.DataFrame(l)				# Transform the read txt into a Pandas df
										# df: column0 = number of sol
										#     column1 = solution array [0,0,4,0,12,0,...0,0,17,0]
										#     column2 = best values of fcost and fdist
	solutions = df0.iloc[:,1]			# Saves into the Variable "solutions" the second column of the df
	solutions = solutions.transpose()	# Transform the line into a column
	df = pd.DataFrame(solutions.apply(pd.Series)).transpose() # Create a df containing a solution in each column
	
	with open(External_Results_Folder + "lookup.txt") as f: # open lookup.txt
		li = [literal_eval(line) for line in f.readlines()] # read and save each line of the file into the variable "li"
	
	df2 = pd.DataFrame(li) # transform the lookup into a Pandas df
	
	joined_dfs = df.join(df2, rsuffix='locations') # join df and df2

	joined_dfs.columns = [str(col) for col in joined_dfs.columns] # transform the column name data type into "str"
	
	N_of_solutions = len(joined_dfs.columns) - 2 # joined_dfs has 1 column for each solution plus 2 columns for the coordinates
	
	sol_df_list = [] # list that contains all the solutions dataframes
	
	for i in range(N_of_solutions):
		a = joined_dfs[joined_dfs[str(i)]>0]
		a = a.loc[:,['0locations','1locations']]
		a.columns = ['X', 'Y']
		sol_df_list.append(a)
	
	# Create a shapefile folder:
	New_Shp_Folder = Results_Folder+'Shapefiles'
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_Shp_Folder): os.makedirs(New_Shp_Folder)
		
	for n, s in enumerate(sol_df_list,1):
		gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in s[['X', 'Y']].values.tolist()],crs={'init':'epsg:19971'})
		gdf.to_file(New_Shp_Folder + '/solution'+ str(n) +'.shp')
"""	
	
def Extract_ParetoFront_and_Plot(Solutions, Norm, External_Results_Folder, Results_Folder, Data_Folder):
	# Function which takes the list of solutions output by the GA and extracts
	# the MOPOs as well as the PF between pairs of solutions before outputting
	# them into a csv file and plotting them.
	
	# Extract a length of the objectives from the first solution
	No_Obj = len(Solutions[0][2])
	
	# Create a folder within the result folder for raster datafiles
	# Raster_Results_Folder = Results_Folder+'Rasters/'    
	# if not os.path.exists(Raster_Results_Folder): os.makedirs(Raster_Results_Folder) 
	
	# EXTRACT MOPOS  

	# Create a list of all the obj functions
	Obj_Functions = range(0,No_Obj)
	
	# Extract the Pareto optimal set for MOPOs    
	MOPOs = NonDom_Sort.Sort(Solutions, Obj_Functions)
	Save_Pareto_Set(MOPOs, Obj_Functions, Norm, Results_Folder)
	Save_to_CSV(MOPOs, Obj_Functions, Results_Folder, Norm)    
	
	# Extract the PF sets between     
	PF_Comb_list = list(combinations(Obj_Functions, 2))
	
	' Write the solutions list to a csv' 
	Save_to_CSV(Solutions, Obj_Functions, Norm, Results_Folder)    
	
	for PF in PF_Comb_list:
		PO_PF = NonDom_Sort.Sort(Solutions, PF)
		
		# print 'The length of Pareto Front between ', Get_String(PF[0]),' and ',Get_String(PF[1]),' is ',len(PO_PF)
		
		Save_Pareto_Set(PO_PF, PF, Norm, Results_Folder)
		Plot(PO_PF, MOPOs, Solutions, PF[0], PF[1], Results_Folder, Norm)
		Save_to_CSV(PO_PF, PF, Norm, Results_Folder)
		
		# Writing solutions of the Pareto front to rasters
		# Write_Rasters(PO_PF, Raster_Results_Folder, Data_Folder, External_Results_Folder)
	# WRITE THE RESULTING RASTER FILES
	
	
def Extract_Generation_Pareto_Fronts(Generations,MinMax_list, Results_Folder, Data_Folder, External_Results_Folder):
	# Module intended to generate the Pareto front of each generation of the
	# Genetic Algorithm run to display how the front converges. So idea is to
	# generate a new folder for the generations within the results folder, 
	# then a folder for each generation and save the PF in CSVs
	
	# Define a new folder within the results folder to hold the 
	   
	Generations_Results_Folder = Results_Folder+'Generations/'
	
	# Generate the generations results folder 
	if not os.path.exists(Generations_Results_Folder): os.makedirs(Generations_Results_Folder)
	# for gen in Generations:
		# print "tl"
		# print len(gen)
	# Now for each generations
	Gen_count = 0
	for Gen in Generations:
		# print "Processing Generation ", Gen_count
		sys.stdout.write(" Processing generation %d \r" %Gen_count)
		# Define the folder to save each particular generation to and create it
		Each_Generations_Results_Folder = Generations_Results_Folder+'Generation_'+str(Gen_count)+'/'
		if not os.path.exists(Each_Generations_Results_Folder): os.makedirs(Each_Generations_Results_Folder)

		# Normalise the generation
		Normalised_Gen = Normalise_Solutions(MinMax_list, Gen)
		
		# Extract the Pareto fronts of this generations and save to csvs
		Extract_ParetoFront_and_Plot(Gen, False, External_Results_Folder, Each_Generations_Results_Folder, Data_Folder)
		
		# Repeat the process with the normalised solutions
		Extract_ParetoFront_and_Plot(Normalised_Gen, True, External_Results_Folder, Each_Generations_Results_Folder, Data_Folder)
		
		# Create shapefiles
		# Create_sol_shapefile(Each_Generations_Results_Folder, External_Results_Folder)
		
		Gen_count += 1
		
def Save_dist_Clinics_Meshblocks(External_Results_Folder, Results_Folder):
	# # This function saves in a .csv file the distances between each Meshblock centroid and the assigned clinic
	
	# First of all transform the solutions from a .txt to a pandas dataframe
	l = []
	with open(Results_Folder + "PO_Set_fdist_fcost_.txt") as f: # Result folder should already be updated to the Run folder
		for line in f.readlines():
			l.append(literal_eval(line))
	
	df0 =  pd.DataFrame(l)				# Transform the read txt into a Pandas df
										# df: column0 = number of sol
										#     column1 = solution array [0,13,0,0,4,0,...0,0,5,0]
										#     column2 = best values of fcost and fdist
	solutions = df0.iloc[:,1]			# Saves into the Variable "solutions" the second columns of the df
	solutions = solutions.transpose()	# Transform the line into a column
	df = pd.DataFrame(solutions.apply(pd.Series)).transpose() # Create a df containing a solution in each column
	
	with open(External_Results_Folder + "lookup.txt") as f: # open lookup.txt
		li = [literal_eval(line) for line in f.readlines()] # read and save each line of the file into the variable "li"
	
	df2 = pd.DataFrame(li) # transform the lookup into a Pandas df
	
	joined_dfs = df.join(df2, rsuffix='locations') # join df and df2

	joined_dfs.columns = [str(col) for col in joined_dfs.columns] # transform the column name data type into "str"
	
	N_of_solutions = len(joined_dfs.columns) - 2 # joined_dfs has 1 column for each solution plus 2 columns for the coordinates
	
	sol_df_list = [] # list that contains all the solutions dataframes
	
	for i in range(N_of_solutions):
		a = joined_dfs[joined_dfs[str(i)]>0]
		a = a.loc[:,['0locations','1locations']]
		a.columns = ['X_Clinic', 'Y_Clinic']
		sol_df_list.append(a)
	
	
	dist_dict_file = 'Dictionary_Avail_Mshbl'
	
	# Creates the dataframe from csv file:
	avail_mshbl_df = pd.read_csv(External_Results_Folder+dist_dict_file, names=['X_Clinic','Y_Clinic','X_Mshbl','Y_Mshbl','TravTime'])
	
	# Create a Distance csv folder:
	New_Dist_Folder = Results_Folder+'Distance_csv/'
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_Dist_Folder): os.makedirs(New_Dist_Folder)
	
	for n, s in enumerate(sol_df_list,1):
		# Merge the solution df with the df containing the distances from targets
		Selected_points_trgt_df = avail_mshbl_df.merge(s, on=['X_Clinic','Y_Clinic'])
		
		# sort data frame on targets' coordinates and dist value:
		Selected_points_trgt_df.sort_values(['X_Mshbl','Y_Mshbl','TravTime'], ascending =[True, True, True], inplace=True) 
		
		# keep the first row of each Meshblock (which contains the closest clinic):
		Selected_points_trgt_df.drop_duplicates(subset=['X_Mshbl','Y_Mshbl'], keep='first', inplace=True) 
	
		# Save the dictionary into a .csv file
		Selected_points_trgt_df.to_csv(New_Dist_Folder+'Dist_clinics_mshbl_sol'+str(n), columns = ('X_Clinic','Y_Clinic','X_Mshbl','Y_Mshbl','TravTime'),header=('X_Clinic','Y_Clinic','X_Mshbl','Y_Mshbl','TravTime_[min]'), index=False)

def Generate_Solutions_Tables(Results_Folder, External_Results_Folder):
	# This function creates single csv files each containing a single solution and its coordinates
	
	l = []
	with open(Results_Folder + "PO_Set_fdist_fcost_.txt") as f: # Result folder should already be updated to the Run folder
		for line in f.readlines():
			l.append(literal_eval(line))
	
	df0 = pd.DataFrame(l)				# Transform the read txt into a Pandas df
										# df: column0 = number of sol
										#     column1 = solution array [0,0,4,0,12,0,...0,0,17,0]
										#     column2 = best values of fcost and fdist
	sol_ID_numbers = df0.iloc[:,0]		# Saves into a variable the solutions IDs
	solutions = df0.iloc[:,1]			# Saves into the Variable "solutions" the second column of the df
	solutions = solutions.transpose()	# Transform the line into a column
	df = pd.DataFrame(solutions.apply(pd.Series)).transpose() # Create a df containing a solution in each column
	
	sol_ID_list = sol_ID_numbers.values # save the values into a list
	
	# add the prefix "sol_" to the numbers of the solutions because I can't have a 0 or a 1 since I need them
	# keys of a dataframe
	newID_sol_list = []
	for l in sol_ID_list:
		new_ID = "sol_"+str(l)
		newID_sol_list.append(new_ID)
	
	df.columns = newID_sol_list
	
	with open(External_Results_Folder + "lookup.txt") as f: # open lookup.txt
		li = [literal_eval(line) for line in f.readlines()] # read and save each line of the file into the variable "li"
	
	df2 = pd.DataFrame(li) # transform the lookup into a Pandas df
	
	joined_dfs = df.join(df2, rsuffix='locations') # join df and df2

	joined_dfs.columns = [str(col) for col in joined_dfs.columns] # transform the column name data type into "str"
	
	# Rename columns
	joined_dfs=joined_dfs.rename(columns = {'0':'X'})
	joined_dfs=joined_dfs.rename(columns = {'1':'Y'})
	
	# Create a shapefile folder:
	New_SolTables_Folder = Results_Folder+'Solutions_tables/'
	# Check it doesn't already exists and then makes it a new directory
	if not os.path.exists(New_SolTables_Folder): os.makedirs(New_SolTables_Folder)
	
	for i in newID_sol_list:
		a = joined_dfs[joined_dfs[str(i)]>0]
		a = a.loc[:,[str(i),'X','Y']]
		a.to_csv(New_SolTables_Folder+'Solution_'+str(i)+'.csv', header=False, index=False)
		