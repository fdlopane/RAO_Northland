# -*- coding: utf-8 -*-
"""
Initialise
Initialisation module

Functions:
	+ Generate_Lookup
		@ It opens a specific availability raster.
		@ Then, by running through it, identifies sites which are developable.
        @ These are saved into a lookup list, which is saved as txt and returned.
	+ Generate_ClinicsPlan
		@ Function creates a Clinics_Plan based the availability Lookup.
		@ It generates a solution with a random number of Clinics
		@ between 1 and Clinics_Max
	+ Generate_Proposed_Sites
		@ Function that takes a Clinics plan as an input and returns
		@ a list of the coordinates of the proposed sites
"""
# Import modules:

"""
# sys.path.append:
# works as it will search the anaconda directory (which is the default python installation)
# for any libraries you try to import, and when it doesn’t find them, it will then search the python27 directory
# as we have added this as a path to look in. The only downside to this is that is a version of a library is in
# the anaconda directory, it will use that, rather than the version you have coped over.
import sys
sys.path.append('C:/Python27/Lib/site-packages')
"""

import random as rndm
import numpy as np
import pandas as pd
import rasterIO
import time
import os.path
import sys
import fiona
import math
from osgeo import gdal
from shutil import copyfile
import scipy
# from scipy import misc
from pathlib import Path

# import Calc_fdist_values_preprocess_02 as network_analysis_preproess
import Constraints


def Generate_Lookup(Data_Folder, Results_Folder, shapefilefile):
	# Creates a list of all the available sites.
		
	time_lookup_start = time.asctime()
	print "Generating lookup, starts at: " , time_lookup_start
	
	# if centroids shapefile exists, use it, otherwise create it:
	if os.path.isfile(os.path.join(Data_Folder, 'Available_centroids.shp')):
		available_centroids_file = 'Available_centroids.shp'
	else:
		print "Available centroid shape file is missing. You must create it!"
		quit()
	
	centroids = fiona.open(Data_Folder+shapefilefile)
	
	# Create a list of all the points:
	Lookup = []
	for p in centroids:
		x_p = p['properties']['X']
		y_p = p['properties']['Y']
		# x_p = p['properties']['POINT_X']
		# y_p = p['properties']['POINT_Y']
		
		x_p = int(math.trunc(x_p)) # convert to integer rounding down
		y_p = int(math.trunc(y_p))
		
		Lookup.append([(x_p),(y_p)])
	
	#save to a txt file in Results Folder so other modules can load it
	np.savetxt(os.path.join(Results_Folder, "lookup.txt"), Lookup, delimiter=',', newline='\n') 	

	time_lookup_end = time.asctime()
	print "Lookup created at:" , time_lookup_end
	print
	return Lookup


def Generate_Lookup_local(Data_Folder, Results_Folder):
	# file is the basic availabilty raster
	File = "Available.tif"
	
	# Import the Availability Raster to identify sites for the Lookup_local
	file_pointer        = rasterIO.opengdalraster(Data_Folder+File)  
	Availability_Raster = rasterIO.readrasterband(file_pointer,1) 
	driver, XSize, YSize, proj_wkt, geo_t_params = rasterIO.readrastermeta(file_pointer)
	
	Lookup_local = [] # Array to hold the location of available sites   
	
	# Investigate for all x and y combinations in the file
	for x in range(0,XSize):
		for y in range(0,YSize):
			# If the yx location yeilds a available site
			if Availability_Raster[y,x]==1:
				# format it and append it to the Lookup_local list
				yx = (y,x)
				Lookup_local.append(yx)
	
	# save to a txt file so other modules can load it
	np.savetxt(os.path.join(Results_Folder, "lookup_local.txt"), Lookup_local, delimiter=',', newline='\n')      

	# return the Lookup_local list
	return Lookup_local


def Generate_DevPlan(Development_Plan, Data_Folder, External_Results_Folder):
     # Produce development plan  
     
     file_pointer = rasterIO.opengdalraster(Data_Folder+'Empty_plan.tif')  
     DevPlan      = np.double(np.copy(rasterIO.readrasterband(file_pointer,1)))
     
     file_pointer = rasterIO.opengdalraster(Data_Folder+'border.tif')     
     Boundary     = np.double(np.copy(rasterIO.readrasterband(file_pointer,1)))
     
     # Upload the Lookup_local table from the generated file.
     Lookup_local = (np.loadtxt(External_Results_Folder+"lookup_local.txt",delimiter=",")).tolist()
     
     # for each site in the Development Plan (with the same length as the Lookup_local)
     for t in range(0, len(Lookup_local)-1):
         
         # find the sites yx location
         j, i = tuple(Lookup_local[t])
         
         # Add the proposed development to the development plan
         #print Development_Plan[j]
         DevPlan[int(j), int(i)] = Development_Plan[t]
         
     # multiplying it to try stop it being square in the raster
     return np.multiply(DevPlan, Boundary)


def Generate_ClinicsPlan(No_Available, Clinics_Max, Clinics_Min, Results_Folder, Data_Folder, Ratio_Doc_Pop, Min_Doc_factor, Max_Doc_factor):
	# Function creates a Clinics_Plan based the availability Lookup length
	# generates a solution with a random number of Clinics between Clinics_Min and Clinics_Max
	# Assigns a number of doctors proportional to the optimal/ideal number of doctors (in relation to number
	# patients living within a 20min drive
	
	# Upload the lookup table from the generated file.
	Lookup_list = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
    
	# Upload the file containing information about number of patients living within a 20min drive from each available cell.
	N_patients_list_csv = Data_Folder + "Av_20min_ServiceArea"
	N_Patients_list_df = pd.read_csv(N_patients_list_csv, names=['Object_ID','X','Y','N_patients'])
	
	# Transform the lookup list into a pandas DataFrame
	Lookup = pd.DataFrame(Lookup_list, columns=['X','Y'])
	
	# Join the two DataFrames:
	Lookup_w_patients = Lookup.merge(N_Patients_list_df, how='inner', on=['X','Y'], suffixes=('', '_2'))
	
	# Handles preventing the code hanging
	check_sum   = 0 # stores the previous Agg_Cl to indicate if its changed
	check_count = 0 # counts the number of iterations the Agg_Cl remains unchanged

	Clinics_Plan = [0]*No_Available # Stores proposed Clinics sites
	Agg_Cl       = 0				# Aggregate number of Clinics assigned 
    
	# Assign a random number of Clinics between 1 and Clinics_Max:
	Number_of_Clinics = rndm.randint(Clinics_Min, Clinics_Max)
	
	# Ensure enough Clinics are assigned
	while Agg_Cl < Number_of_Clinics:
	
		# Select a random available site
		j = rndm.randint(0,No_Available-1)
		ji =  Lookup_list[j]
		ji[0] = int(math.trunc(ji[0]))
		ji[1] = int(math.trunc(ji[1]))
		
		# Check if the site hasn't already been designated.
		if Clinics_Plan[j] == 0:
			# Assign a random number of doctors in the range [0.5*Opt_doc , 1.5*Opt_doc]
			Selected_row = Lookup_w_patients.loc[(Lookup_w_patients['X'] == ji[0]) & (Lookup_w_patients['Y'] == ji[1])] # row of the df that contains coords and pop
			N_patients = int(Selected_row['N_patients'])
			
			Opt_doc = max(1, int(round(N_patients*Ratio_Doc_Pop))) # optimal number of doctors
			
			min_doct = int(max(1, round(Min_Doc_factor*Opt_doc))) # min number of doctors
			max_doct = int(round(Max_Doc_factor*Opt_doc))		  # max number of doctors
			
			Clinics_Plan[j] = rndm.randint(min_doct, max_doct)
			Agg_Cl += 1
			
			# Prevents the code hanging:
			if check_sum == Agg_Cl:
				# if the Agg_Cl was the same on the last iteration the count is increased
				check_count += 1
			else:
				# if Agg_Cl is different reset count and take the new Agg_Cl 
				check_count = 0
				check_sum = Agg_Cl
			
			# If the iteration has gone through with no change return false   
			if check_count > 100000:
				print "Caught hanging in Generate_ClinicsPlan"           
				return False
			
	# sys.stdout.write("Generation of Clinics plan Completed.\r")
	
	if sum(Clinics_Plan) == 0:
		raise ValueError('Sum of Clinics plan = 0. No sites allocated in Generate_ClinicsPlan function.')
	
	# print Clinics_Plan
	# print "Generate_ClinicsPlan"
	
	return Clinics_Plan


def Generate_Proposed_Sites(Clinics_Plan, Results_Folder):
	# Returns a list of the coordinates of the proposed sites and the number of doctors assigned to each site.
	Lookup = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup

	# print sum(Clinics_Plan)
	# print "Generate_Proposed_Sites"

	# if sum(Clinics_Plan) == 0:
		# print sum(Clinics_Plan)
		# print "Generate_Proposed_Sites"
		# raise ValueError('Sum of Clinics plan = 0. When called in Generate_Proposed_Sites function.')
	
	Proposed_Sites_List = []
	# for each site in the Clinics Plan (with the same length as the lookup)
	for j in range(0, len(Lookup)-1):
		if Clinics_Plan[j] > 0:
			# find the sites yx location
			ji =  Lookup[j]
			ji[0] = int(math.trunc(ji[0]))
			ji[1] = int(math.trunc(ji[1]))
			# ji = tuple(ji)
			Proposed_Sites_List.append([ji[0], ji[1], Clinics_Plan[j]])
	# if len(Proposed_Sites_List) == 0:
		# raise ValueError('Length of Proposed sites list = 0 Length of Clinics plan = ', sum(Clinics_Plan))
		
	return Proposed_Sites_List


def Generate_Min_Max_Doc(No_Available, Results_Folder, Data_Folder, Ratio_Doc_Pop, Min_Doc_factor, Max_Doc_factor):
	# Function the creates a list of the minimum number of doctors for each available cell 
	# and a list of the maximum number of doctors for each available cell 
	
	# Upload the lookup table from the generated file.
	Lookup_list = (np.loadtxt(os.path.join(Results_Folder, "lookup.txt"),dtype='int',delimiter=",")).tolist() # reads the content of the .txt and saves it in Lookup
    
	# Upload the file containing information about number of patients living within a 20min drive from each available cell.
	N_patients_list_csv = Data_Folder + "Av_20min_ServiceArea"
	N_Patients_list_df = pd.read_csv(N_patients_list_csv, names=['Object_ID','X','Y','N_patients'])
	
	# Transform the lookup list into a pandas DataFrame
	Lookup = pd.DataFrame(Lookup_list, columns=['X','Y'])
	
	# Join the two DataFrames:
	Lookup_w_patients = Lookup.merge(N_Patients_list_df, how='inner', on=['X','Y'], suffixes=('', '_2'))
	
	Min_doctors_Plan = [0]*No_Available # Stores proposed Clinics sites
	Max_doctors_Plan = [0]*No_Available # Stores proposed Clinics sites
	
	for j in range(No_Available):
	
		ji =  Lookup_list[j]
		ji[0] = int(math.trunc(ji[0]))
		ji[1] = int(math.trunc(ji[1]))
		
		# Assign the min and max number of doctors for each cell
		Selected_row = Lookup_w_patients.loc[(Lookup_w_patients['X'] == ji[0]) & (Lookup_w_patients['Y'] == ji[1])] # row of the df that contains coords and pop
		N_patients = int(Selected_row['N_patients'])
		
		Opt_doc = max(1, int(round(N_patients*Ratio_Doc_Pop))) # optimal number of doctors
		
		min_doct = int(max(1, round(Min_Doc_factor*Opt_doc))) # min number of doctors
		max_doct = int(round(Max_Doc_factor*Opt_doc))		  # max number of doctors
		
		Min_doctors_Plan[j] = min_doct
		Max_doctors_Plan[j] = max_doct
	
	return Min_doctors_Plan, Max_doctors_Plan