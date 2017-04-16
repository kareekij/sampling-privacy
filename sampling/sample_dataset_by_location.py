# -*- coding: utf-8 -*-

"""
Script for the sampling algorithm
"""

from __future__ import division, print_function
import numpy as np
import networkx as nx
import csv
import sys
import random
import operator
import time
import math
import argparse

import oracle
import query
import log


import community
import _mylib
import Queue
from scipy import stats

from sklearn import linear_model

import pickle

def read_profile(sel_location):
	print('Reading Profile..')
	profile = []
	with open('./data/soc-pokec-profiles.txt', 'rb') as f:
		reader = csv.reader(f, delimiter='\t')
		for row in reader:
			id = row[0]
			location = row[4]
			if location == sel_location:
				profiles.append(id)
	return profile

if __name__ == '__main__':
	fname = './data/soc-pokec-relationships.txt'
	sel_location = 'bratislavsky kraj, bratislava - karlova ves'

	G = _mylib.read_file(fname)
	profiles = read_profile(sel_location)

	print(len(profiles), len(set(profiles)))

	s = G.subgraph(profiles)
	n = s.number_of_nodes()
	e = s.number_of_edges()
	print(n,e)
	pickle_path = './data/pokec-loc-' + str(n) + '.pickle'
	pickle.dump(s, open(pickle_path, 'wb'))


