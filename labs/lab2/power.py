import numpy as np
from random import choices

def permutation_test(data1, data2, reps):

	tobs = np.mean(data2) - np.mean(data1)
	data_all = np.hstack((data1, data2))
	size = len(data1)
	
	perm = []
	for _ in range(reps):
		perm_data = np.random.permutation(data_all)
		pdata1 = perm_data[:size]
		pdata2 = perm_data[-size:]
		tperm = np.mean(pdata2) - np.mean(pdata1)
		perm.append(tperm > tobs)
	
	return np.mean(perm)

def power(sample1, sample2, reps, size, alpha):

	pvalues = []
	for _ in range(reps):
		data1 = choices(sample1, k=size)
		data2 = choices(sample2, k=size)
		pval = permutation_test(data1, data2, 19000)
		pvalues.append(pval)
	
	return np.mean(np.array(pvalues) < (1.0-alpha))

if __name__ == "__main__":

	old = np.array([0,0,0,0,0,0,1,0,0,1,0])
	new = np.array([1,0,0,1,1,1,0,0,0,1,0])

	p = power(old, new, 10, 11, 0.05)
	print(f"Power = {p}")
