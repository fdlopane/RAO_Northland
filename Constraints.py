# -*- coding: utf-8 -*-
"""
Module Containing a series of constraint handling for the genetic algorithm
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

import numpy as np
import os
import random as rndm

"""

def Remove(orig_tuple, element_to_remove):
	# Function which is called once a child in the offspring is found to
	# exceed the maximum number of doctors or be lower than the minimum or higher than the maximum
	lst = list(orig_tuple)
	lst.remove(element_to_remove)
	# return the offspring array with the element removed
	return tuple(lst)
"""

"""
"""
def Check_N_Doctors(Min_doc, Max_doc, Results_Folder):
	# Fuction that checks if the number of doctors is within the boundaries
	
	def decCheckBounds(func):
		def wrapCheckBounds(*args, **kargs):

			# Extract the offspring solutions
			offsprings = func(*args, **kargs)
			strt_len = len(offsprings)
			# Extract each of the children from the offspring
			for child in offsprings:
				num_doctors = sum(child)

				if num_doctors < Min_doc or num_doctors > Max_doc:
					# if a clinics plan doesn't fall between the min and max it is removed from the offspring
					offsprings.remove(child)
					# print('removed child')

			end_len = len(offsprings)

			# Calculate the number of solutions retained after the constraint
			per_retained = float(100 * end_len / strt_len)

			# Load the previous list of retaintion rates and add new retention
			Retained_list = np.loadtxt(Results_Folder+"N_doctors_Constraint.txt", delimiter=",")
			Updated_Retained_list = np.append(Retained_list, per_retained)
			# Save the updated list
			np.savetxt(Results_Folder+"N_doctors_Constraint.txt", Updated_Retained_list, delimiter=',', newline='\n',fmt="%i")

			# print '% of solutions retained after Total Dwellings Constraint', per_retained
			return offsprings
		return wrapCheckBounds
	return decCheckBounds

def Mutation_constraint(Min_doctors_Plan, Max_doctors_Plan):
	# Function that checks if afetr the mutation the number of doctors for each cell is still within the acceptable range
	
	def decCheckBounds(func):
		def wrapCheckBounds(*args, **kargs):

			# Extract the offspring solutions
			offsprings = func(*args, **kargs)
			strt_len = len(offsprings)
			# Extract each of the children from the offspring
			for child in offsprings:
				for i, av in enumerate(child):
					if av !=0:
						if av < Min_doctors_Plan[i] or av > Max_doctors_Plan[i]:
							# if the number of doctors is less than min or more than max, change it with a random
							# number between min and max
							child[i] = rndm.randint(Min_doctors_Plan[i], Max_doctors_Plan[i])
							# print 'modified child, from: ', av, " to: ", child[i], " [range: (",Min_doctors_Plan[i], ",", Max_doctors_Plan[i], ") ]"
			return offsprings
		return wrapCheckBounds
	return decCheckBounds

