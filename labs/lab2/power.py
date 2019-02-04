import numpy as np
from random import choices

def permutation_test(data1, data2, size, reps):

	#tobs = np.mean(data2) - np.mean(data1)
	# correction!
	tobs = size

	data_all = np.hstack((data1, data2))
	data_size = len(data1)
	
	perm = []
	for _ in range(reps):
		perm_data = np.random.permutation(data_all)
		pdata1 = perm_data[:data_size]
		pdata2 = perm_data[-data_size:]
		tperm = np.mean(pdata2) - np.mean(pdata1)
		perm.append(tperm > tobs)
	
	# return counts instead of the p-value
	return np.mean(perm)

def power(sample1, sample2, reps, size, alpha):

	pvalues = []
	for _ in range(reps):
		data1 = choices(sample1, k=len(sample1))
		data2 = choices(sample2, k=len(sample2))
		pval = permutation_test(data1, data2, size, 19000)
		pvalues.append(pval)
	
	return np.mean(np.array(pvalues) < (1.0-alpha))

if __name__ == "__main__":

	old = np.array([0,0,0,0,0,0,1,0,0,1,0])
	new = np.array([1,0,0,1,1,1,0,0,0,1,0])

	alpha = 0.95
	p = power(old, new, 10, 11, alpha)
	print(f"Power = {p}")
