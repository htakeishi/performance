# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
import csv

from matplotlib import pyplot as plt

x = np.linspace(-5,5,1000)
x_discrete = np.arange(-5,5)

SIZE = 1000
processig_times = stats.norm.rvs(loc=3, scale=2, size=SIZE )

mu = 4.0
intervals = stats.poisson.rvs(mu, size=SIZE )

TIMES = 10000
MAX = 60
process_time_distribution = [0]*MAX

for j in range(TIMES):
    debt = 0.0

    for i in range(SIZE):
        debt = max(0, debt + (max(0, processig_times[i]) - intervals[i]) ) # 負債は 0 以上の値にする。
        p    = max(0,processig_times[i]) + debt
        if (p < MAX):
            process_time_distribution[int(p)] += 1

hist_sum = sum(int(i) for i in process_time_distribution)
process_time_distribution = list( map(lambda x: x/float(hist_sum), process_time_distribution ) )

for i in range(MAX-1):
    process_time_distribution[i+1] = process_time_distribution[i]+process_time_distribution[i+1]

output_array = np.array(process_time_distribution)
np.savetxt("out".csv", output_array, delimiter=",")
