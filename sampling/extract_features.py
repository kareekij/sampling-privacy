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
import pickle
import _mylib

class ExtractFeatures(object):
	def __init__(self, sample_pickle):
		super(ExtractFeatures, self).__init__()
		self.sample_graph = self.read_pickle(sample_pickle)



	def read_pickle(self, fname):
		return pickle.load(open(fname, 'rb'))

	def get_start_node(self):
		start_node = _mylib.get_members_from_com(1, nx.get_node_attributes(self.sample_graph, 'is_start'))
		return start_node

	def get_target_node(self):
		target_node = _mylib.get_members_from_com(1, nx.get_node_attributes(self.sample_graph, 'is_target'))
		return target_node

	def get_profile(self):
		return nx.get_node_attributes(self.sample_graph, 'public_rand')

	def get_adjacency_matrix(self):
		return nx.to_numpy_matrix(self.sample_graph)

	def get_node_order(self):
		return self.sample_graph.nodes()

	def get_label(self, type):
		return nx.get_node_attributes(self.sample_graph, str(type))

	# Below are functions for getting graph properties
	def get_degree(self):
		return self.sample_graph.degree()

	def get_clustering_coefficient(self):
		return self.sample_graph.clustering()




if __name__ == '__main__':
	fname = './output/sample_500_610.pickle'

	e = ExtractFeatures(fname)

	print(e.get_start_node())
	a = (e.get_node_order())
	b = (e.get_label('gender'))

	print(a[:10])
	print(b.values()[:10])
	#print(e.get_node_order())

	# G = read_pickle(fname)
	#
	# start_node = _mylib.get_members_from_com(1,nx.get_node_attributes(G,'is_start'))
	# target_node = _mylib.get_members_from_com(1, nx.get_node_attributes(G, 'is_target'))
	# label = nx.get_node_attributes(G, 'gender')
	#
	# print(label)
	#
	# A = nx.to_numpy_matrix(G)
	# cc = nx.clustering(G)
	# deg = G.degree()