import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
from plot_utils import load_data

profile_name = r"manyant.step.shower"

problem_range = range(100, 250 + 1, 10)

variants = [ "sequential", "manyant", ]
#variants = [ "sequential", "manyant", ]
problems = [f"ESC63s{k}.sop" for k in problem_range]

files = [
	f"evaluation/{profile_name}.profile.csv",
	"evaluation/sequential.step.shower-2.profile.csv"
]

data = load_data([
    f"evaluation/{profile_name}.profile.csv",
	"evaluation/sequential.step.shower-2.profile.csv",
])

adva = load_data(files, measurement="adva")
adva = {v: [np.mean(adva[(v, p)]) for p in problems] for v in variants}

upda = load_data(files, measurement="upda")
upda = {v: [np.mean(upda[(v, p)]) for p in problems] for v in variants}

eval = load_data(files, measurement="eval")
eval = {v: [np.mean(eval[(v, p)]) for p in problems] for v in variants}


data = {v: [np.mean(data[(v, p)]) / 1000.0 for p in problems] for v in variants}

x = np.arange(len(problems))
width = 0.2

fig, ax = plt.subplots(figsize=(6, 4))

# for i, (attr, values) in enumerate(data.items()):
# 	offset = width * i
# 	#r = ax.bar(x + offset + 0.2, values, width, label=attr, zorder=3)
# 	ax.plot(values, label=attr, zorder=3)

for i, ((var, aval), uval, eval) in enumerate(zip(adva.items(), upda.values(), eval.values())):
	offset = width * i
	#r = ax.bar(x + offset + 0.2, values, width, label=attr, zorder=3)
	#ax.plot(values, label=attr, zorder=3)
	# ax.stackplot(
	# 	np.arange(len(problem_range)), aval, uval, eval,
	# 	labels=("adva", "upda", "eval"),
	# 	alpha=0.3)
	alpha = 0.6 if i == 0 else 1
	dashes = [4, 4] if i == 0 else (None, None)
	ax.plot(aval, label="adva" if i == 1 else None, color="tab:blue", dashes=dashes, alpha=alpha)
	ax.plot(eval, label="eval" if i == 1 else None, color="tab:orange", dashes=dashes, alpha=alpha)
	ax.plot(uval, label="upda" if i == 1 else None, color="tab:green", dashes=dashes, alpha=alpha)

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
#ax.set_ylabel("Ausführungszeit (s)")
ax.set_ylabel("Ausführungszeit (ms)")
ax.set_ylim(ymin=2e-4, ymax=1e4)
ax.ticklabel_format(axis="y", useMathText=True)
ax.xaxis.set_major_formatter(get_problem_size)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis="y", which="minor", color="0.7")
fig.tight_layout()
ax.set_yscale("log")

solid_line = Line2D([0], [0], dashes=(None, None), color="black")
dashed_line = Line2D([0], [0], dashes=[4, 4], color="black")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [solid_line, dashed_line], labels + ["manyant", "sequential"], loc="upper left", ncols=2)
#ax.legend(loc="upper left", ncols=3)



#fig.savefig(f"evaluation/figures/{profile_name}.png", dpi=200)
plt.show()
