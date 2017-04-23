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

starting_node = -1

class UndirectedSingleLayer(object):
	"""
	Class for Expansion-Densification sammling
	"""
	def __init__(self, query, budget=100, exp_type='oracle', dataset=None, logfile=None, log_int=10):
		super(UndirectedSingleLayer, self).__init__()
		self._budget = budget 			# Total budget for sampling
		self._query = query 			# Query object
		self._dataset = dataset 		# Name of the dataset used; Used for logging and caching
		self._logfile = logfile 		# Name of the file to write log to
		self._exp_type = exp_type
		self._log_interval = log_int
		self._stage = None
		self._one_cost = 0
		self._one_gain = 0

		self._cost = 0 					# Keep track of the cost spent
		self._sample = {'edges': set(), 'nodes':\
		 {'close':set(), 'open':set()}}
		self._wt_exp = 1 				# Weight of the expansion score
		self._wt_den = 3 				# Weight of the densification score
		self._score_den_list = [] 			# List to store densification scores; used only for logging
		self._score_exp_list = [] 			# List to store expansion scores; used only for logging
		self._new_nodes = []			# List of the new nodes observed at each densification
		self._cumulative_new_nodes = [] # Cummulated new nodes
		self._exp_cut_off = 50			# Numbor of expansion candidates
		self._den_cut_off = 100 			# Number of densificaiton candidates
		self._sample_graph = nx.Graph() # The sample graph
		self._nodes_observed_count = [] # Number of nodes observed in each iteration
		self._avg_deg = 0.
		self._med_deg = 0.
		self._cost_spent = []
		self._nodes_return = []
		self._exp_count = 0
		self._densi_count = 0
		self._track_obs_nodes = []
		self._track_cost = []
		self._track = {}
		self._track_edges = {}
		self._track_cc = {}
		self._track_new_nodes = []

		self._track_k = []
		self._track_open= []

		self._tmp = 0
		self._avg = {'unobs':[], 'close':[], 'open':[]}

		self._percentile = 90
		self._sd = []

		self._line_1 = []
		self._line_2 = []

		self._X = []
		self._Y = []



	def random_sampling(self):

		current = starting_node

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		# TODO: Densification switch criteria
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			den_nodes = sub_sample['nodes']['open']

			# Randomly pick node
			current = random.choice(list(den_nodes))

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current)
			self._count_new_nodes(nodes, current)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _updateSample(self, sub_sample):
		"""
		Update the sample with the sub sample

		Args:
			sub_sample (dict) -- The sub sample dict
		"""
		self._sample['edges'].update(sub_sample['edges'])
		self._sample['nodes']['close'].update(sub_sample['nodes']['close'])
		self._sample['nodes']['open'] = self._sample['nodes']['open'].difference(\
		 sub_sample['nodes']['close'])
		self._sample['nodes']['open'].update(sub_sample['nodes']['open'])


		nodes_count = self._sample_graph.number_of_nodes()
		# TODO: Calculate mean and median of close nodes degree, MIGHT CHANGE !
		#degree = self._sample_graph.degree().values()
		degree = self._sample_graph.degree(self._sample['nodes']['close']).values()
		self._avg_deg = np.mean(np.array(degree))
		self._med_deg = np.median(np.array(degree))
		# print( " Degree avg: {} , med: {}".format(self._avg_deg, self._med_deg))

	def _updateSubSample(self, sub_sample, nodes, edges, candidate):
		"""
		Update the sub sample with new nodes aned edges

		Args:
			sub_sample (dict) -- The sub sample to update
			nodes (list[str]) -- The open nodes
			edges (list[(str,str)]) -- The new edges
			candidate (str) -- The new open node
		Return:
			dict -- The updated sub sample
		"""
		try:

			sub_sample['edges'].update(edges)
			sub_sample['nodes']['close'].add(candidate)
			sub_sample['nodes']['open'].remove(candidate)
			sub_sample['nodes']['open'].update(\
				nodes.difference(sub_sample['nodes']['close'])\
				.difference(self._sample['nodes']['close']))
		except KeyError as e:
			print('		subsample update:', e)


		return sub_sample

	def _bfs(self):

		"""
		Collect the initial nodes through bfs

		Args:
			None
		Return:
			None
		"""

		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}

		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]
		private_count = 0

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c = self._query.neighbors(current)

			if current != target_node:
				self._increment_cost(c)

			if len(nodes) != 0:
				self._count_new_nodes(nodes, current)

				for e in edges:
					self._sample_graph.add_edge(e[0], e[1])

				if target_node in set(nodes):
					print(' -- Target found ! --')
			else:
				#print('		Private user queried')
				private_count += 1

			# Remove the current node from queue
			queue.remove(current)

			# Update queue
			if len(nodes) != 0:
				queue += list(nodes.difference(sub_sample['nodes']['close']).difference(sub_sample['nodes']['open']))
				queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)
			#print('Queue: ', len(queue))



		# Updat the sample with the sub sample
		self._updateSample(sub_sample)
		print("Reach {} private users, {} total users found".format(private_count, self._sample_graph.number_of_nodes()))

	def _random_walk(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)


		private_count = 0

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			if current_node != target_node:
				self._increment_cost(c)

			if len(nodes) != 0:
				self._count_new_nodes(nodes, current_node)

				for e in edges:
					self._sample_graph.add_edge(e[0], e[1])

				if target_node in set(nodes):
					print(' -- Target found ! --')
			else:
				private_count += 1

			if current_node in sub_sample['nodes']['open']:
				# Update the sub sample
				sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Candidate nodes are the (open) neighbors of current node
			candidates = list(set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
			print('		Current: {}, {} neighbors, {} open'.format(current_node, len(nodes), len(candidates)))

			while len(candidates) == 0:
				if len(nodes) != 0:
					current_node = random.choice(list(nodes))
				else:
					current_node = random.choice(list(self._sample_graph.nodes()))

				r = random.uniform(0,1)
				if r < 0.15:
					current_node = random.choice(list(self._sample_graph.nodes()))
					print(' JUMP!', self._cost)

				print('	 Walking .. current node', current_node)
				# Query the neighbors of current
				nodes, edges, c = self._query.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
				#print("RW: getting stuck")

			current_node = random.choice(candidates)



		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _count_new_nodes(self, nodes, current):
		current_nodes = self._sample_graph.nodes()
		new_nodes = set(nodes).difference(current_nodes)
		c = len(new_nodes)
		self._track_new_nodes.append(current)

	def _max_obs_deg(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			degree_observed = self._sample_graph.degree(candidates)
			degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
			current_node = degree_observed_sorted[0][0]



		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_score(self):
		candidates = self._sample['nodes']['open']

		current_node = starting_node

		# End

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(
					self._sample['nodes']['close']))

			current_node = self._pick_from_close(candidates)
			#current_node = self._cal_score(candidates)
			#score_sorted = _mylib.sortDictByValues(score, reverse=True)

			#current_node = score_sorted[0][0]

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _pick_from_close(self, candidates):
		close_nodes = set(self._sample_graph.nodes()) - set(candidates)
		deg = self._sample_graph.degree(close_nodes)
		print(' pick from ', len(deg), len(close_nodes))

		if len(deg) <= 10:
			return random.choice(list(candidates))

		sorted_deg = _mylib.sortDictByValues(deg,reverse=True)
		new_cand = set()

		for t in sorted_deg[:10]:
			n = t[0]
			nbs = self._sample_graph.neighbors(n)
			open_nb = set(nbs).intersection(candidates)
			print(len(open_nb))
			new_cand.update(open_nb)

		if len(new_cand) == 0:
			new_cand = candidates

		return random.choice(list(new_cand))

		# open_nb = set()
		# while len(open_nb) == 0:
		# 	print('		Deg size', len(deg), len(open_nb))
		# 	node = _mylib.pickMaxValueFromDict(deg)
		# 	nbs = self._sample_graph.neighbors(node)
		# 	open_nb = set(nbs).intersection(candidates)
		# 	deg = _mylib.removekey(deg, node)

		#return random.choice(list(open_nb))


	def _cal_score(self, candidates):
		open_nodes = candidates
		close_nodes = set(self._sample_graph.nodes()) - set(candidates)

		degree_cand = self._sample_graph.degree(candidates)
		degree_close = self._sample_graph.degree(close_nodes)
		degree_avg_close = np.mean(np.array(degree_close.values()))

		pos_list = {}
		neg_list = {}
		for candidate in candidates:
			deg = degree_cand[candidate]
			deg_diff = deg - degree_avg_close
			print('			deg',deg)
			if deg_diff >= 0 :
				pos_list[candidate] = deg_diff
			else:
				neg_list[candidate] = deg_diff

		print('Total: {} -- Pos{} Neg{}'.format(len(candidates), len(pos_list),len(neg_list)))

		if len(pos_list) != 0:
			print('		[Cal S] Positive list', degree_avg_close)
			n = _mylib.pickMaxValueFromDict(pos_list)
		else:
			print('		[Cal S] Negative list', degree_avg_close)
			n = _mylib.pickMaxValueFromDict(neg_list)
			#n = random.choice(list(neg_list.keys()))


		return n

	def _pick_max_score(self, score):
		max_val = max(score.values())
		np_score = np.array(score.values())
		max_idx = np.argmax(np_score)
		node = score.keys()[max_idx]
		#print(np_score)
		print(' max-score pick:', score[node])
		return node

	def _learn_model(self):
		candidates = self._sample['nodes']['open']
		degree_observed = self._sample_graph.degree(candidates)
		degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
		current_node = degree_observed_sorted[0][0]

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)
			self._get_training(sub_sample, nodes, edges, current_node)
			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(
					self._sample['nodes']['close']))
			# Start picking a new node to query
			if self._cost <= 50:
				degree_observed = self._sample_graph.degree(candidates)
				degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
				current_node = degree_observed_sorted[0][0]
			else:
				y = np.array(self._Y)
				#cut_off = np.median(self._sample_graph.degree(sub_sample['nodes']['close']).values())
				# cut_off = np.mean(y)
				#cut_off = 0.1

				#y[np.array(self._Y) >= cut_off] = 1
				#y[np.array(self._Y) < cut_off] = 0


				model = self._build_model(y)
				testing = self._get_testing(candidates)

				#print(np.shape(self._X))
				print(np.shape(np.array(testing)))


				#candidates = random.sample(candidates,5)
				#candidates_deg = self._sample_graph.degree(candidates)
				#candidates_cc = nx.clustering(self._sample_graph,candidates)

				#A = np.array([candidates_deg.values(), candidates_cc.values()]).transpose()
				#print(len(candidates))
				#print(A)

				y_predict = model.predict(testing)
				# print(y_predict)
				#
				max_val = (max(y_predict))
				print('max val', max_val)
				y_idx = np.where(y_predict == max_val)[0]

				#
				#
				# # y_idx = np.nonzero(y_predict)[0]
				# #
				if len(y_idx) != 0:
					pick = random.choice(y_idx)
					#print('pick index', pick, len(candidates))
					current_node = list(candidates)[pick]
				# else:
				# 	print('all zeros')
				# 	current_node = random.choice(candidates)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _build_model(self, y):
		np_X = np.array(self._X)

		logistic = linear_model.LogisticRegression()
		logistic.fit(np_X, y)

		return logistic

	def _test_algo(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			all_closed_nodes = sub_sample['nodes']['close'].union(self._sample['nodes']['close'])

			# Candidate nodes are the (open) neighbors of current node.
			candidates = list(set(nodes).difference(all_closed_nodes))

			TELEPORT_P = 0.1
			# If not randomly pick one neighbors and look at its neighbors instead
			while len(candidates) == 0:
				r = random.uniform(0,1)
				if r <= TELEPORT_P:
					current_node = self._get_max_open_nbs_node(all_closed_nodes)
				else:
					current_node = random.choice(list(nodes))

				# Query the neighbors of current
				#nodes, edges, c = self._query.neighbors(current_node)
				nodes = self._sample_graph.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(set(nodes).difference(all_closed_nodes))
				print("Test Algo: Move to neighbor")

			r = random.uniform(0, 1)
			if r <= TELEPORT_P:
				current_node = self._get_max_open_nbs_node(all_closed_nodes)
			else:
				current_node = random.choice(candidates)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _get_max_open_nbs_node(self, close_n):

		count_n = {}
		for n in close_n:
			nodes = self._sample_graph.neighbors(n)
			candidates = list(set(nodes).difference(close_n))
			count_n[n] = len(candidates)

		max_count = max(count_n.values())
		print('Max count', max_count)

		candidates = _mylib.get_members_from_com(max_count, count_n)
		return random.choice(candidates)


	def _getDenNodes(self, nodes):
		"""
		Generate a list of best densification nodes based on clustering coeff

		Only the number of nodes with count highest clustering
		coefficient are to be considered for densification

		Args:
			nodes(list([str])) -- Open nodes list
		Return:
			list[str] -- List of self._den_cut_off with highest clustering coeff
		"""
		if len(nodes) > self._den_cut_off:
			# Get clustering coefficients of the nodes
			cc = nx.clustering(self._sample_graph, nodes)
			# Sort nodes by clustering coefficient in descending order
			max_val = cc.values()
			candidates = _mylib.get_members_from_com(max_val,cc)

			if len(candidates) > self._den_cut_off:
				return random.sample(candidates, self._den_cut_off)
			else:
				return candidates
			# cc = sorted(cc, key=cc.get, reverse=True)
			# return cc[:self._den_cut_off]
		else:
			return list(nodes)

	def _getExpNodes(self):
		"""
		Generate a list of best expansion nodes based on clustering coeff

		Only the number of nubmer of nodes with count lowest clustering
		coefficient are to be considered for expansion

		Considers all the open nodes. Not just from latest subsample

		Args:
			None
		Return:
			list[str] -- The self._exp_cut_off nodes with lowest clustering coeff
		"""
		if len(self._sample['nodes']['open']) > 0:
			# Clustering coeff of the open nodes in the sample
			cc = nx.clustering(self._sample_graph, self._sample['nodes']['open'])
			# Sort the nodes by clustering coeff in ascending order
			cc = sorted(cc, key=cc.get, reverse=False)
			return cc[:self._exp_cut_off]
			#return cc
		else:
			print('	*No open nodes')
			return list(nodes)

	def _track_cost_spent_return(self):
		cur_return = len(self._sample['nodes']['close']) + len(self._sample['nodes']['open'])

		if len(self._cost_spent) == 0:
			prev_cost = 0
			prev_return = 0
		else:
			prev_cost = self._cost_spent[-1]
			prev_return = self._nodes_return[-1]

		cost_used = self._cost - prev_cost
		cost_used = self._cost - prev_cost
		node_gain = cur_return - prev_return

		self._nodes_return.append(node_gain)
		self._cost_spent.append(cost_used)

		#self._nodes_return.append(cur_return - prev_return)

		# self._cost_spent.append(self._cost - prev_cost)

		#self._cost_spent.append([self._stage, self._cost, cur_return])
		#self._nodes_return.append(cur_return)

	def _increment_cost(self, cost ):
		self._cost += cost


		if  int(self._cost) % self._log_interval == 0:
			obs_nodes = self._sample_graph.number_of_nodes()
			obs_edges = self._sample_graph.number_of_edges()

			c = int(self._cost)

			self._track[c] = obs_nodes
			self._track_edges[c] = obs_edges

			self._track_k.append(self._tmp)
			self._track_open.append(len(self._sample['nodes']['open']))


			nodes_count = self._sample_graph.number_of_nodes()
			edges_count = self._sample_graph.number_of_edges()

			self._line_1.append(nodes_count)
			self._line_2.append(edges_count)

	def generate(self):
			"""
			The main method that calls all the other methods
			"""
			#self._bfs()
			current_list = []

			sample_G = None
			is_break = False


			current = starting_node


			if self._exp_type == 'random':
				self.random_sampling()
			elif self._exp_type == 'rw':
				self._random_walk()
			elif self._exp_type == 'bfs':
				print('bfs')
				self._bfs()

			print('			Budget spent: {}/{}'.format(self._cost, self._budget))

			print('			Number of nodes \t Close: {} \t Open: {}'.format( \
				len(self._sample['nodes']['close']), \
				len(self._sample['nodes']['open'])))

			"""
			repititions = 0
			for x in self._oracle._communities_selected:
				repititions += self._oracle._communities_selected[x]
			repititions = repititions - len(self._oracle._communities_selected)
			print(self._oracle._communities_selected, len(self._oracle._communities_selected), repititions)
			"""

def Logging(sample):
	# Keep the results in file
	track_sort = _mylib.sortDictByKeys(sample._track)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]


	log.log_new_nodes(log_file, dataset, type, obs_track, cost_track, budget, bfs_budget)

	print("---- DONE! ----")
	print("	# Exp: {}, # Den: {}".format(sample._exp_count, sample._densi_count))
	print('	Nodes in S: ', sample._sample_graph.number_of_nodes())
	#print('	Clustering Coeff =', nx.average_clustering(graph))
	print('-'*15)

def SaveToFile(results_nodes,results_edges, query_order):
	log.save_to_file(log_file_node, results_nodes)
	log.save_to_file(log_file_edge, results_edges)
	log.save_to_file(log_file_order, query_order)

def Append_Log(sample, type):
	track_sort = _mylib.sortDictByKeys(sample._track)
	track_edges_sort = _mylib.sortDictByKeys(sample._track_edges)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]
	obs_edges_track = [x[1] for x in track_edges_sort]

	if type not in Log_result:
		Log_result[type] = obs_track
		Log_result_edges[type] = obs_edges_track
	else:
		Log_result[type] += (obs_track)
		Log_result_edges[type] += (obs_edges_track)

	if type not in Log_result_nn:
		Log_result_nn[type] = sample._track_new_nodes
	else:
		Log_result_nn[type] += sample._track_new_nodes

	return cost_track

def read_pickle(fname):
	return pickle.load(open(fname, 'rb'))

def read_profile():
	print('Reading Profile..')
	profile = {}
	with open('./data/soc-pokec-profiles.txt', 'rb') as f:
		reader = csv.reader(f, delimiter='\t')
		for row in reader:
			profile[row[0]] = np.array(row)
			# [[1,3,4,7,11,13]]
	return profile

def randomTargetNodes(G, pri_users, gender_dict, sel_gender):
	deg = G.degree(pri_users)
	deg_k = deg.keys()
	deg_v = deg.values()

	DEG_T = 5
	DEG_MEAN = np.mean(np.array(G.degree().values()))
	DEG_MED = np.median(np.array(G.degree().values()))
	print("  Degree T= {} Mean: {} Med: {}".format(DEG_T, DEG_MEAN, DEG_MED))

	sel_target = random.choice(pri_users)
	sel_deg = deg[sel_target]
	sel_target_gender = gender_dict[sel_target]

	while not (sel_deg >= DEG_T and int(sel_target_gender) == int(sel_gender)):
		sel_target = random.choice(pri_users)
		sel_deg = deg[sel_target]
		sel_target_gender = gender_dict[sel_target]
#		print('Test', sel_gender, sel_target_gender, sel_deg, (sel_deg >= DEG_T), ((sel_deg >= DEG_T and sel_target_gender == sel_gender)))

	#print( "Random Target {} deg {}, gender {}".format(sel_target, sel_deg, sel_target_gender))

	return sel_target, sel_target_gender





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-task', help='Type of sampling', default='undirected_single')
	parser.add_argument('-fname', help='Edgelist file', type=str, default='./data-input/pokec-loc-7654_attr_20.pickle')
	parser.add_argument('-budget', help='Total budget', type=int, default=1000)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-log', help='Log file', default='./log/')
	parser.add_argument('-experiment', help='# of experiment', default=10)
	parser.add_argument('-log_interval', help='# of budget interval for logging', type=int, default=10)
	parser.add_argument('-mode', help='mode', type=int, default=2)
	parser.add_argument('-delimiter', help='csv delimiter', type=str, default=None)
	parser.add_argument('-n', help='n', type=str, default=20)

	args = parser.parse_args()

	print(args)

	fname = args.fname
	budget = args.budget
	dataset = args.dataset
	log_file = args.log
	log_interval = args.log_interval
	mode = args.mode
	delimeter = args.delimiter
	fold_n = args.n


	if mode == 1:
		exp_list = ['bfs']
	elif mode == 2:
		exp_list = ['rw']


	print(exp_list)
	Log_result = {}
	Log_result_edges = {}
	Log_result_nn = {}

	if dataset == None:
		f = fname.split('.')[1].split('/')[-1]
		dataset = f

	G = read_pickle(fname)
	graph = G

	print('Original: # nodes', G.number_of_nodes())
	print('Original: # edges', G.number_of_edges())



	query = query.UndirectedSingleLayer(graph)

	log_file_node = log_file + dataset + '_n.txt'
	log_file_edge = log_file + dataset + '_e.txt'
	log_file_order = log_file + dataset + '_order.txt'

	node_public = nx.get_node_attributes(graph, 'public_rand')

	pub_users = _mylib.get_members_from_com(1, node_public)
	pri_users = _mylib.get_members_from_com(0, node_public)


	# Setup budget
	n = graph.number_of_nodes()
	if budget == 0:

		budget = int(.10*n)
	print('{} :: Budget set to {} , n={}'.format(dataset, budget, n))

	gender_dict = nx.get_node_attributes(graph, 'gender')
	#group1 = _mylib.get_members_from_com('1', gender_dict)
	#group2 = _mylib.get_members_from_com('0', gender_dict)


	#print(len(group1), len(group2))

	#_mylib.degreeHist(G.degree().values())
	#_mylib.degreeHist_2([G.degree(pub_users).values(), G.degree(pri_users).values()],legend=['public','private'])

	# Sampling starts here
	for i in range(0, int(args.experiment)):
		row = []
		tmp = []
		#targets_sel = [974775, 447622, 240731, 21729, 982749, 642315, 368241, 1420921, 248330, 1035964, 475316, 256294, 593567, 789731]
		isDone = False
		while not isDone:
			sel_gender = (i+1) % 2
			target_node, target_gender = randomTargetNodes(G, pri_users, gender_dict, sel_gender)
			#print(sel_gender, target_gender)
			starting_node = query.randomSameCom(target_node)

			print("	- Public {}, Private {}".format(len(pub_users), len(pri_users)))
			print("	- Starting node: {}".format(starting_node))
			print("	- Target node: {}".format(target_node))

			for type in exp_list:
				sample = UndirectedSingleLayer(query, budget, type, dataset, log, log_interval)

				# if starting_node == -1:
				# 	starting_node = sample._query.randomNode()

				print('[{}] Experiment {} starts at node {}'.format(type, i, starting_node))

				# Getting sample
				sample.generate()
				# End getting sample

				#cost_arr = Append_Log(sample, type)

			if target_node in sample._sample_graph.nodes():
				degree = sample._sample_graph.degree(target_node)
				print('Target {} degree {}'.format(target_node, degree))

				c_nodes = set(sample._sample['nodes']['close'])
				c_nodes.add(target_node)
				sub_g = nx.Graph()
				sub_g = graph.subgraph(c_nodes)

				target_nbs = sample._sample_graph.neighbors(target_node)
				target_nbs_c  = set(target_nbs).intersection(c_nodes)

				print('		Target nbs count {} , {} are closed nodes'.format(len(target_nbs), len(target_nbs_c)))
				print('		Closed Nodes {}, in sample {}'.format(len(c_nodes), sub_g.number_of_nodes() ) )

				is_start = nx.get_node_attributes(sub_g, 'is_start')
				a = _mylib.get_members_from_com(1, nx.get_node_attributes(sub_g, 'is_start'))
				for aa in a:
					is_start[aa] = 0
				is_start[starting_node] = 1

				nx.set_node_attributes(sub_g, 'is_start', is_start)

				is_target = nx.get_node_attributes(sub_g, 'is_target')
				a = _mylib.get_members_from_com(1, nx.get_node_attributes(sub_g, 'is_target'))
				for aa in a:
					is_target[aa] = 0
				is_target[target_node] = 1

				nx.set_node_attributes(sub_g, 'is_target', is_target)

				a = _mylib.get_members_from_com(1, nx.get_node_attributes(sub_g, 'is_start'))
				b = _mylib.get_members_from_com(1, nx.get_node_attributes(sub_g, 'is_target'))
				print('Start', a)
				print('Target', b)

				#sample_fn = './output/private-exp-'+ int(fold_n) + '/sample_' + type + '_' + str(budget) + '_' + str(len(c_nodes)) + '_' + str(len(target_nbs)) + '_' + str(time.time()) + '.pickle'
				#folder_n = './outputRW/budget-exp-private-20/budget-'+str(budget)
				#folder_n = './outputRW/loc-private-20'
				folder_n = './outputRW/private-exp-budget-1000/private-' + str(fold_n)
				sample_fn = folder_n + '/' + str(i) + '_sample_' + type + '_' + str(budget) + '_' + str(len(c_nodes)) + '_' + str(len(target_nbs)) + '_' + str(target_gender) + '.pickle'

				#sample_fn = './output/test'+ str(i) +'.pickle'
				pickle.dump(sub_g, open(sample_fn, 'wb'))

				isDone = True
			else:
				print('Path not found to target {}'.format(target_node))
