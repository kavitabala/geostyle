import numpy as np
from scipy.stats import hmean

class Grouper:
	def __init__(self):
		self.se = 3
		self.pr = 3

	def uniq_subsets(self, s):
		u = set()
		for x in s:
			t = []
			for y in x:
				y = list(y)
				y.sort()
				t.append(tuple(y))
			t.sort()
			u.add(tuple(t))
		return list(u)


	def k_subset(self, s, k):
		if k == len(s):
			return (tuple([(x,) for x in s]),)
		k_subs = []
		for i in range(len(s)):
			partials = self.k_subset(s[:i] + s[i + 1:], k)
			for partial in partials:
				for p in range(len(partial)):
					k_subs.append(partial[:p] + (partial[p] + (s[i],),) + partial[p + 1:])
		return k_subs


	def find_ind_val(self, outliers, scores):
		val = 0
		cons = 15
		if len(outliers) > 1:
			vals = []
			for i in range(1, len(outliers)):
				if outliers[i]-outliers[i-1] <= 52+self.se and outliers[i]-outliers[i-1] >= 52-self.se:
					vals.append(((np.abs(outliers[i]-outliers[i-1]-52))+cons)/(cons+self.se))
				elif outliers[i]-outliers[i-1] <= self.pr:
					vals.append((outliers[i]-outliers[i-1]-1+cons)/(cons+self.pr))
				else:
					vals.append(np.inf)
			val = np.mean(vals)
			# val += 1
			val = val*hmean(scores)
		else:
			val = hmean(scores)
		return val

	def get_best_partition(self, outliers, scores):
		# enumerate all subsets and find the one with minimum cost
		outliers = list(outliers)
		min_val = np.inf
		min_group = None
		min_costs = None
		for i in range(1, len(outliers)+1):
			subsets = self.uniq_subsets(self.k_subset(outliers, i))
			for subset in subsets:
				vals = []
				for ssubset in subset:
					vals.append(self.find_ind_val(ssubset, [scores[outliers.index(tmp)] for tmp in ssubset]))
				val = np.mean(vals)
				if val < min_val:
					min_val = val
					min_group = subset
					min_costs = vals
		return min_group, min_costs
