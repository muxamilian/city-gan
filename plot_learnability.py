#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

fake_europe = [0.06026827490318019, 0.08011957230081633, 0.06301058311126612, 0.08695892001037428]
fake_europe_stds = [0.11637461050072387, 0.12426689608377378, 0.1059825495589539, 0.13042403576853834]

real_europe = [0.604732492098643, 0.5450528988902806, 0.5602732886457524, 0.5226749280476942]
real_europe_stds = [0.3255070518478199, 0.3104947989381465, 0.3147982032895511, 0.3110836051728744]

real_all = [0.6004441868097256, 0.3748713832953477, 0.8013372300479804, 0.8033370512762646, 0.6995585179306238]
real_all_stds = [0.3175287082815527, 0.28044010552489607, 0.2510797117891816, 0.25818860668865573, 0.2802453081348096]

fake_all = [0.111933680299515, 0.051533134962373876, 0.17702940586658314, 0.1753962784992703, 0.1490221186235391]
fake_all_stds = [0.16631454558374795, 0.10218487444366127, 0.22607917095896043, 0.23703448226372714, 0.1801273714384119]


# good_color = matplotlib.colors.to_rgb("xkcd:darkgreen")
# bad_color = matplotlib.colors.to_rgb("xkcd:dark red")

good_color = "green"
bad_color = "red"

def plot(means1, means2, stds1, stds2, labels):

		means1 = np.array(means1)
		means2 = np.array(means2)

		n = len(means1)
		ind = np.arange(n)    # the x locations for the groups
		width = 0.35       # the width of the bars: can also be len(x) sequence

		plt.figure(figsize=(5,3))
		p1 = plt.bar(ind, means1, width, color=bad_color)#, yerr=stds1)
		p2 = plt.bar(ind, means2-means1, width, bottom=means1, color=good_color)#, yerr=stds2)

		plt.ylabel('Probability')
		# plt.title('Scores by group and gender')
		plt.xticks(ind, labels)
		plt.yticks(np.arange(0, 1.1, 0.1))
		plt.legend((p2[0], p1[0]), ('real', 'fake'))

		plt.show()

plot(fake_europe, real_europe, fake_europe_stds, real_europe_stds, ["Amsterdam", "Manhattan", "Paris", "Vienna"])
plot(fake_all, real_all, fake_all_stds, real_all_stds, ["Amsterdam", "D.C.", "Florence", "Las Vegas", "New York"])
