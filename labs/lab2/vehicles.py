import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import salaries as sal # To use implemented MAD function

if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')
    print(df.columns)
    print(len(df.index))

    df_clear = df.dropna()
    print(len(df_clear.index))

    # Creating scatterplots
    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df_clear, fit_reg=False)
    
    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, )

    sns_plot.savefig("scaterplot.png", bbox_inches='tight')
    sns_plot.savefig("scaterplot.pdf", bbox_inches='tight')

    old_fleet = df['Current fleet']
    new_fleet = df['New Fleet'].dropna()

    print("Current fleet:")
    print((("Mean: %f") % (np.mean(old_fleet))))
    print((("Median: %f") % (np.median(old_fleet))))
    print((("Var: %f") % (np.var(old_fleet))))
    print((("std: %f") % (np.std(old_fleet))))
    print((("MAD: %f") % (sal.mad(old_fleet))))

    plt.clf()
    sns_plot2 = sns.distplot(old_fleet, bins=20, kde=False, rug=True).get_figure()

    axes = plt.gca()
    axes.set_xlabel('Miles Per Gallon')
    axes.set_ylabel('Cars count')

    sns_plot2.savefig("histogram_old.png", bbox_inches='tight')
    sns_plot2.savefig("histogram_old.pdf", bbox_inches='tight')

    print("New fleet:")
    print((("Mean: %f") % (np.mean(new_fleet))))
    print((("Median: %f") % (np.median(new_fleet))))
    print((("Var: %f") % (np.var(new_fleet))))
    print((("std: %f") % (np.std(new_fleet))))
    print((("MAD: %f") % (sal.mad(new_fleet))))

    plt.clf()
    sns_plot2 = sns.distplot(new_fleet, bins=20, kde=False, rug=True).get_figure()

    axes = plt.gca()
    axes.set_xlabel('Miles Per Gallon')
    axes.set_ylabel('Cars count')

    sns_plot2.savefig("histogram_new.png", bbox_inches='tight')
    sns_plot2.savefig("histogram_new.pdf", bbox_inches='tight')
