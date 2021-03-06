import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
from random import choices


def bootstrap(sample, sample_size, iterations, ci):

	new_samples = [choices(sample, k=sample_size) for _ in range(iterations)]

	data_mean = np.mean(new_samples)

	iter_mean = np.mean(new_samples, axis=1)

	lower_q = (100.0-ci)/2.0
	lower = np.percentile(iter_mean, lower_q)

	upper_q = ci+lower_q
	upper = np.percentile(iter_mean, upper_q)

	return data_mean, lower, upper

def process_data(data, output):
	
	boots = []
	for i in range(100, 100000, 1000):
		boot = bootstrap(data, data.shape[0], i, 95.0)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig(f"bootstrap_confidence_{output}.png", bbox_inches='tight')
	sns_plot.savefig(f"bootstrap_confidence_{output}.pdf", bbox_inches='tight')

if __name__ == "__main__":

	df = pd.read_csv('./salaries.csv')
	data = df.values.T[1]
	
	process_data(data, "salaries")

	#Exercise 2 (vehicles)

	df_veh = pd.read_csv('./vehicles.csv')

	old_vehs = df_veh['Current fleet'].values
	new_vehs = df_veh['New Fleet'].dropna().values

	process_data(old_vehs, "old_fleet")
	process_data(new_vehs, "new_fleet")