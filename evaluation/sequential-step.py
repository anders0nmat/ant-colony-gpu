import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator
import csv
import re
from pathlib import Path
import numpy as np
import math
from plot_utils import load_data

profile_name = r"sequential.step.shower-2"

problem_range = range(100, 250 + 1, 10)

variants = [ "sequential" ]
problems = [f"ESC63s{k}.sop" for k in problem_range]

data = load_data(f"evaluation/{profile_name}.profile.csv")
data = {v: [np.mean(data[(v, p)]) / 1000.0 for p in problems] for v in variants}

x = np.arange(len(problems))
width = 0.2

fig, ax = plt.subplots(figsize=(6, 4))

for i, (attr, values) in enumerate(data.items()):
	offset = width * i
	#ax.bar(x + offset + 0.2, values, width, label=attr, zorder=3)
	ax.plot(values, label=attr)



#ax.set_xticks(
#	x + width * 0.5 * (len(variants) + 1),
#	[re.match(r"ESC63s(\d+)\.sop", p).group(1) for p in problems])
def get_problem_size(x, pos):
	off = problem_range.start
	step = problem_range.step
	return int(off + x * step)

ax.set_xlabel("Problemgröße")
ax.grid(visible=True, axis="y", which="both", zorder=0)
ax.grid(which="minor", axis="y", color="0.95")
ax.grid(visible=True, axis="x", which="both", zorder=0, color="0.9")

ax.set_xlim(0, len(problems) - 1)
ax.set_ylabel("Ausführungszeit (s)")
ax.set_ylim(ymin=0)
ax.ticklabel_format(axis="y", useMathText=True)
ax.xaxis.set_major_formatter(get_problem_size)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(loc="upper left", ncols=3)
ax.tick_params(axis="y", which="minor", color="0.7")
fig.tight_layout()

#fig.savefig(f"evaluation/figures/{profile_name}.png", dpi=200)
plt.show()
