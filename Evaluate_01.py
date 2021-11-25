# -*- coding: utf-8 -*-
"""
Objectives currently optimised include:
    1. Travel times
    2. Costs

"""

"""
# sys.path.append:
# works as it will search the anaconda directory (which is the default python installation)
# for any libraries you try to import, and when it doesn’t find them, it will then search the python27 directory
# as we have added this as a path to look in. The only downside to this is that is a version of a library is in
# the anaconda directory, it will use that, rather than the version you have coped over.
import sys
sys.path.append('C:/Python27/Lib/site-packages')
"""

# program_name = "Evaluate_01"

import os
# os.system('cls')  # clears screen

import time
# start_time = time.asctime()

# print "Program: " , program_name
# print "Starts at: " , start_time
# print
# print "Importing modules..."
import fiona
import pandas as pd
import copy
import sys
import Initialise_01 as Init
import math
import numpy as np
# print "Modules imported."
# print


"""
"""
def Calc_fcost(Clinics_Plan, Wc, Wd):
	# Cost function:
	# fcost = Wc * N_Clinics + Wd * N_Doctors
		
	# Determine the number of clinics:
	clinics = []
	for c in Clinics_Plan:
		if c == 0:
			clinics.append(c)
		else:
			c = c/c
			clinics.append(c)
	tot_n_clinics = sum(clinics)
	
	# Determine number of doctors:
	tot_n_doctors = sum(Clinics_Plan)
	
	fcost = (Wc * tot_n_clinics) + (Wd * tot_n_doctors)
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if sum(Clinics_Plan) == 0:
		fcost = 300 # Meaningless very high number
	
	return int(fcost)

"""
"""
def Calc_fcost_constraint(Clinics_Plan, Wc, Wd, Min_doc, Max_doc):
	# Cost function:
	# fcost = Wc * N_Clinics + Wd * N_Doctors
		
	# Determine the number of clinics:
	clinics = []
	for c in Clinics_Plan:
		if c == 0:
			clinics.append(c)
		else:
			c = c/c
			clinics.append(c)
	tot_n_clinics = sum(clinics)
	
	# Determine number of doctors:
	tot_n_doctors = sum(Clinics_Plan)
	
	fcost = (Wc * tot_n_clinics) + (Wd * tot_n_doctors)
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		pass
	else:
		fcost = 300 # Meaningless very high number
	
	return int(fcost)

"""
"""
def Calc_fcost_Diff_TOT_Opt_constraint(Clinics_Plan, Optimal_ratio, Tot_population, Min_doc, Max_doc):
	# Cost function:
	# fcost = 1 / |Optimal_ratio - Solution_ratio| Calculated on the whole region
	
	# Determine number of doctors:
	tot_n_doctors = sum(Clinics_Plan)
	
	fcost = abs(1/Optimal_ratio - (Tot_population/tot_n_doctors)) + 1 # I don't want it to be 0 because when I normalize I don't want to divide by 0
	
	# print fcost
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		pass
	else:
		fcost = 3000 # Meaningless very high number
	
	return int(fcost)

"""
"""
def Calc_fcost_Diff_Opt_constraint(Clinics_Plan, Optimal_ratio, Tot_population, Min_doc, Max_doc, Results_Folder, Data_Folder, Ratio_Doc_Pop):
	# Cost function:
	# fcost = [90°percentile of |1/Optimal_ratio - 1/Solution_ratio| for each clinic] + 1
	
	# Upload the lookup table from the generated file.
	Lookup_list = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
    
	# Upload the file containing information about number of patients living within a 20min drive from each available cell.
	N_patients_list_csv = Data_Folder + "Av_20min_ServiceArea"
	N_Patients_list_df = pd.read_csv(N_patients_list_csv, names=['Object_ID','X','Y','N_patients'])
	
	# Transform the lookup list into a pandas DataFrame
	Lookup = pd.DataFrame(Lookup_list, columns=['X','Y'])
	
	# Join the two DataFrames:
	Lookup_w_patients = Lookup.merge(N_Patients_list_df, how='inner', on=['X','Y'], suffixes=('', '_2'))
	
	diff_list = [] # list that contains all the differences between the opt n of doctors and the assigned one for aech clinic
	
	for c, i in enumerate(Clinics_Plan):
		
		ji =  Lookup_list[c-1] # c is the counter of "enumerate" and starts from 1
		ji[0] = int(math.trunc(ji[0]))
		ji[1] = int(math.trunc(ji[1]))
		
		if i != 0:
			Selected_row = Lookup_w_patients.loc[(Lookup_w_patients['X'] == ji[0]) & (Lookup_w_patients['Y'] == ji[1])] # row of the df that contains coords and pop
			N_patients = int(Selected_row['N_patients'])
			
			# Opt_doc = max(1, int(round(N_patients*Ratio_Doc_Pop))) # optimal number of doctors
			
			diff_opt = abs(1/Ratio_Doc_Pop - N_patients/i)
			diff_list.append(diff_opt)
			
	# Transfor the list into a dataframe
	diff_df = pd.DataFrame(diff_list)
	
	# Calculate the 90° quantile
	Quantile_90 = diff_df.quantile(0.9, interpolation='linear') # 0.9 = 90° percentile
	
	fcost = Quantile_90 + 1 # add +1 becasuse I don't want it to be 0 (even if I pick the "perfect" plan)
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		# print(sum(Clinics_Plan))
		pass
	else:
		# print "__",sum(Clinics_Plan)
		fcost = 3000 # Meaningless very high number
	
	return int(fcost)

"""
"""
def Calc_fdist(Results_Folder, Proposed_Sites, X_quantile):
	# Function that calculates the average of the 90° percentile of the travel time from
	# each MeshBlock centroid to the closest clinic
	
	# determine the closest clinic for each Meshblock centroid
	dist_dict_file = '/Dictionary_Avail_Mshbl' # File that contains: X_avail, Y_avail, X_mshblck, Y_mshblck, Dist, Pop
	
	# Create the dataframe from csv file:
	mshblck_clinics_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Msh','Y_Msh','TravTime', 'Pop'])
	
	# Create a dataframe containing the coordinates of the proposed sites:
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail', 'Pop'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = mshblck_clinics_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on MeshBlock centroids' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Msh','Y_Msh','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Keep the first row of each MeshBlock centroid (which contains the closest clinic):
	Selected_points_trgt_df.drop_duplicates(subset=['X_Msh','Y_Msh'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the clinics with ascending TravTime:
	Selected_points_trgt_df.sort_values(['X_Avail','Y_Avail','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and TravTime
	C_TT_df = Selected_points_trgt_df[['X_Avail','Y_Avail','TravTime']]
	
	# Calculate the XX quantile (XX determined by initial data)
	Quantile_df = C_TT_df.groupby(['X_Avail', 'Y_Avail']).quantile(X_quantile, interpolation='lower')
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if len(Proposed_Sites) == 0:
		fdist = 150 # Meaningless very high number
	else:
		q_mean = Quantile_df.mean()
		fdist = q_mean[0] # The mean of the quantiles of each clinic
	
	return fdist

"""
"""
def Calc_fdist_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan):
# Function that calculates the average of the 90° percentile of the travel time from
	# each MeshBlock centroid to the closest clinic
	
	# determine the closest clinic for each Meshblock centroid
	dist_dict_file = '/Dictionary_Avail_Mshbl' # File that contains: X_avail, Y_avail, X_mshblck, Y_mshblck, Dist, Pop
	
	# Create the dataframe from csv file:
	mshblck_clinics_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Msh','Y_Msh','TravTime', 'Pop'])
	
	# Create a dataframe containing the coordinates of the proposed sites:
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail', 'Pop'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = mshblck_clinics_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on MeshBlock centroids' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Msh','Y_Msh','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Keep the first row of each MeshBlock centroid (which contains the closest clinic):
	Selected_points_trgt_df.drop_duplicates(subset=['X_Msh','Y_Msh'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the clinics with ascending TravTime:
	Selected_points_trgt_df.sort_values(['X_Avail','Y_Avail','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and TravTime
	C_TT_df = Selected_points_trgt_df[['X_Avail','Y_Avail','TravTime']]
	
	# Calculate the XX quantile (XX determined by initial data)
	Quantile_df = C_TT_df.groupby(['X_Avail', 'Y_Avail']).quantile(X_quantile, interpolation='lower')
	
	# The mean of the quantiles of each clinic
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		q_mean = Quantile_df.mean()
		fdist = q_mean[0] # The mean of the quantiles of each clinic
	else:
		fdist = 150 # Meaningless high number		
	
	return fdist

"""
"""
def Calc_fdist_average_dist_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan):
	# Function that calculates the average travel time from
	# each MeshBlock centroid to the closest clinic
	
	# determine the closest clinic for each Meshblock centroid
	dist_dict_file = '/Dictionary_Avail_Mshbl' # File that contains: X_avail, Y_avail, X_mshblck, Y_mshblck, Dist, Pop
	
	# Create the dataframe from csv file:
	mshblck_clinics_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Msh','Y_Msh','TravTime', 'Pop'])
	
	# Create a dataframe containing the coordinates of the proposed sites:
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail', 'Pop'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = mshblck_clinics_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on MeshBlock centroids' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Msh','Y_Msh','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Keep the first row of each MeshBlock centroid (which contains the closest clinic):
	Selected_points_trgt_df.drop_duplicates(subset=['X_Msh','Y_Msh'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the clinics with ascending TravTime:
	Selected_points_trgt_df.sort_values(['X_Avail','Y_Avail','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and TravTime
	C_TT_df = Selected_points_trgt_df[['X_Avail','Y_Avail','TravTime']]
	
	# Calculate the mean of the Travel Time column
	TravTime_mean = C_TT_df.loc[:,"TravTime"].mean()
	
	
	# The mean of the quantiles of each clinic
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		fdist = TravTime_mean
	else:
		fdist = 150 # Meaningless high number		
	
	return fdist

"""
"""
def Calc_fdist_threshold_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan, TT_Threshold, W_A, W_T):
	# Function that calculates the sum of average travel time from
	# each MeshBlock centroid to the closest clinic and the average of all the travel times > of a threshold
	# fdist = TT_mean + TT(>Threshold)_mean
	
	# determine the closest clinic for each Meshblock centroid
	dist_dict_file = '/Dictionary_Avail_Mshbl' # File that contains: X_avail, Y_avail, X_mshblck, Y_mshblck, Dist, Pop
	
	# Create the dataframe from csv file:
	mshblck_clinics_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Msh','Y_Msh','TravTime', 'Pop'])
	
	# Create a dataframe containing the coordinates of the proposed sites:
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail', 'Pop'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = mshblck_clinics_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on MeshBlock centroids' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Msh','Y_Msh','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Keep the first row of each MeshBlock centroid (which contains the closest clinic):
	Selected_points_trgt_df.drop_duplicates(subset=['X_Msh','Y_Msh'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the clinics with ascending TravTime:
	Selected_points_trgt_df.sort_values(['X_Avail','Y_Avail','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and TravTime
	C_TT_df = Selected_points_trgt_df[['X_Avail','Y_Avail','TravTime']]
	
	# Calculate the mean of the Travel Time column
	TravTime_mean = C_TT_df.loc[:,"TravTime"].mean()
	
	# Calculate the mean of the travel times > TT_Threshold:
	GreaterThanThreshold_mean = C_TT_df["TravTime"].loc[C_TT_df["TravTime"]>TT_Threshold].mean()
	
	
	# The mean of the quantiles of each clinic
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		fdist = W_A*TravTime_mean + W_T*GreaterThanThreshold_mean
	else:
		fdist = 150 # Meaningless high number		
	
	return fdist

"""
"""
def Calc_fdist_GEUD_constraint(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan, GEUD_power):
	# Function that calculates the "à la Generalised Equivalent Uniform Dose" travel time from
	# each MeshBlock centroid to the closest clinic
	
	# determine the closest clinic for each Meshblock centroid
	dist_dict_file = '/Dictionary_Avail_Mshbl' # File that contains: X_avail, Y_avail, X_mshblck, Y_mshblck, Dist, Pop
	
	# Create the dataframe from csv file:
	mshblck_clinics_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Msh','Y_Msh','TravTime', 'Pop'])
	
	# Create a dataframe containing the coordinates of the proposed sites:
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail', 'Doctors'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = mshblck_clinics_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on MeshBlock centroids' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Msh','Y_Msh','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Keep the first row of each MeshBlock centroid (which contains the closest clinic):
	Selected_points_trgt_df.drop_duplicates(subset=['X_Msh','Y_Msh'], keep='first', inplace=True)
	
	# Sort the DataFrame grouping the clinics with ascending TravTime:
	Selected_points_trgt_df.sort_values(['X_Avail','Y_Avail','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Create a Dataframe with only the coordinates of clinics and TravTime
	# C_TT_df = Selected_points_trgt_df[['X_Avail','Y_Avail','TravTime']]
	
	# Save travel times values into a variable:
	TT_var = Selected_points_trgt_df['TravTime']
	TT_list = TT_var.values # save the values into a list
	
	GEUD = 0
	
	for tt in TT_list:
		GEUD = GEUD + tt**GEUD_power
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		fdist = GEUD**(1.0/GEUD_power)
	else:
		fdist = 1500 # Meaningless high number		
	
	return fdist

"""
"""
def Calc_fdist_weight_pop(Results_Folder, Proposed_Sites, X_quantile, Min_doc, Max_doc, Clinics_Plan):
	# Function that calculates the travel time from each MeshBlock centroid to the closest clinic
	# with a weight proportional to the population served
	
	# determine the closest clinic for each Meshblock centroid
	dist_dict_file = '/Dictionary_Avail_Mshbl' # File that contains: X_avail, Y_avail, X_mshblck, Y_mshblck, Dist, Pop
	
	# Create the dataframe from csv file:
	mshblck_clinics_df = pd.read_csv(Results_Folder+dist_dict_file, names=['X_Avail','Y_Avail','X_Msh','Y_Msh','TravTime', 'Pop'])
	
	# Create a dataframe containing the coordinates of the proposed sites:
	proposed_sites_df = pd.DataFrame.from_records(Proposed_Sites, columns=['X_Avail','Y_Avail', 'Doctors'])
	
	# Merge the DataFrames on the base of the common columns (coordinates of available sites):
	Selected_points_trgt_df = mshblck_clinics_df.merge(proposed_sites_df, on=['X_Avail','Y_Avail'])
	
	# Sort data frame on MeshBlock centroids' coordinates and dist value:
	Selected_points_trgt_df.sort_values(['X_Msh','Y_Msh','TravTime'], ascending =[True, True, True], inplace=True)
	
	# Keep the first row of each MeshBlock centroid (which contains the closest clinic):
	Selected_points_trgt_df.drop_duplicates(subset=['X_Msh','Y_Msh'], keep='first', inplace=True)
	
	# Save travel times values into a variable:
	TT_var = Selected_points_trgt_df['TravTime']
	TT_list = TT_var.values # save the values into a list
	
	# Save population values into a variable:
	pop_var = Selected_points_trgt_df['Pop']
	pop_list = pop_var.values # save the values into a list
	
	fdist_sum = 0
	
	for i in range(len(TT_list)):
		fdist_sum = fdist_sum + (TT_list[i] * pop_list[i])
	
	# If a Clinics plan is empty, the variable Proposed_Sites will be empty.
	# It will be eliminated in the evaluation process, but I need to be able 
	# to evaluate it. So, if len(Proposed_sites)==0 --> assign a very high value to fdist
	if Min_doc < sum(Clinics_Plan) < Max_doc:
		fdist = fdist_sum / sum(pop_list)
	else:
		fdist = 60 # Meaningless high number		
	
	return fdist
