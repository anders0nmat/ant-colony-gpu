import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
from plot_utils import load_data

profile_name = r"constant.step.shower"

#problem_range = range(150, 350 + 1, 10)
problem_range = range(400, 1000 + 1, 100)

variants = [ "neighbor", "constant" ]
problems = [f"ESC63s{k}.sop" for k in problem_range]

files = [
	f"evaluation/{profile_name}.profile.csv",
    "evaluation/neighbor.step.shower.profile.csv"
]

data = load_data(files)

data = {v: [np.mean(data[(v, p)]) / 1000.0 for p in problems] for v in variants}

adva = load_data(files, measurement="adva")
adva = {v: [np.mean(adva[(v, p)]) for p in problems] for v in variants}

upda = load_data(files, measurement="upda")
upda = {v: [np.mean(upda[(v, p)]) for p in problems] for v in variants}

eval = load_data(files, measurement="eval")
eval = {v: [np.mean(eval[(v, p)]) for p in problems] for v in variants}


x = np.arange(len(problems))
width = 0.2

fig, ax = plt.subplots(figsize=(6, 4))

for i, ((attr, values), aval, uval, eval) in enumerate(zip(data.items(), adva.values(), upda.values(), eval.values())):
	offset = width * i
	#r = ax.bar(x + offset + 0.2, values, width, label=attr, zorder=3)
	#ax.plot(values, label=attr, zorder=3)
	# ax.stackplot(
	# 	np.arange(len(problem_range)), aval, uval, eval,
	# 	labels=("adva", "upda", "eval"),
	# 	alpha=0.3)
	alpha = 0.2 if i == 0 else 1
	ax.plot(aval, label="adva" if i == 1 else None, color="tab:blue", alpha=alpha)
	ax.plot(eval, label="eval" if i == 1 else None, color="tab:orange", alpha=alpha)
	ax.plot(uval, label="upda" if i == 1 else None, color="tab:green", alpha=alpha)


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
ax.set_ylabel("Ausführungszeit (ms)")
ax.set_ylim(ymin=1e-2, ymax=1e3)
ax.ticklabel_format(axis="y", useMathText=True)
ax.xaxis.set_major_formatter(get_problem_size)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(loc="upper left")
ax.tick_params(axis="y", which="minor", color="0.7")
fig.tight_layout()
ax.set_yscale("log")

#fig.savefig(f"evaluation/figures/{profile_name}.png", dpi=200)
plt.show()
