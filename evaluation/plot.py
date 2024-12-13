import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import csv
import argparse
from dataclasses import dataclass
from collections import defaultdict
import re

argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("file", nargs="+", help="CSV-file with result data")
argumentParser.add_argument("-w", "--width", type=float, default=0.5, help="Width of bars")
argumentParser.add_argument("-e", "--errors", action="store_true", help="Show error bars")
argumentParser.add_argument("-l", "--log", action="store_true", help="Use logarithmic y-scale")
argumentParser.add_argument("-o", "--output", default=None, help="Automatically export the created ficure to OUTPUT")
argumentParser.add_argument("-t", "--title", default="", help="Set title of figure")

argumentParser.add_argument("--order", action="extend", nargs="+", help="Orders the output groups")

g1 = argumentParser.add_mutually_exclusive_group()
g1.add_argument("-v", "--variant", action="store_true", help="Group by Variant")
g1.add_argument("-p", "--problem", action="store_true", help="Group by Problem")

g2 = argumentParser.add_mutually_exclusive_group()
g2.add_argument("-s", "--sections", action="store_true", help="Show execution sections: adva, eval, upda")
g2.add_argument("-c", "--score", action="store_true", help="Show normalized RPS score instead of execution time")
g2.add_argument("--section", action="append", help="Only show the specified measurement section")

args = argumentParser.parse_args()

def minmax_tuple(t1: tuple[float, float], t2: tuple[float, float]) -> tuple[float, float]:
    return (
        min(t1[0], t2[0]),
        max(t1[1], t2[1]),
    )

@dataclass
class Profile:
    rounds: int
    prep: float
    optr: float
    opts: float
    adva: float
    eval: float
    upda: float
    etc: float = 0
    count: int = 1
    minmax: tuple[float, float] = (0, 0)

    def __post_init__(self):
        self.minmax = (self.optr, self.optr)

    def add(self, other: "Profile"):
        if self.rounds != other.rounds:
            raise ValueError("Profiles have different round count")

        self.prep += other.prep
        self.optr += other.optr
        self.opts += other.opts
        self.adva += other.adva
        self.eval += other.eval
        self.upda += other.upda
        self.minmax = minmax_tuple(self.minmax, other.minmax)
        self.count += other.count

    def average(self):
        self.prep /= self.count
        self.optr /= self.count
        self.opts /= self.count
        self.adva /= self.count
        self.eval /= self.count
        self.upda /= self.count
        self.count = 1
        
    def total(self):
        self.adva *= self.rounds
        self.eval *= self.rounds
        self.upda *= self.rounds

        self.etc = self.optr - self.adva - self.eval - self.upda


variants: dict[tuple[str, str],Profile] = {}

for file in args.file:
    with open(file) as f:
        data = csv.DictReader(f, delimiter=";")
        
        for line in data:
            key = (line["variant"], line["problem"])
            if line["variant"].startswith("#"):
                continue
            profile = Profile(
                rounds=int(line["rounds"]),
                prep=float(line["prep"]),
                optr=float(line["optr"]),
                opts=float(line["opts"]),
                adva=float(line["adva"]),
                eval=float(line["eval"]),
                upda=float(line["upda"])
            )

            if key not in variants:
                variants[key] = profile
            else:
                variants[key].add(profile)        

keys = args.order
if not keys and (args.variant or args.problem):
    keys = []
    all_keys = set()
    available_keys = defaultdict(set)
    for v, p in variants.keys():
        if args.variant:
            available_keys[p].add(v)
            if v not in all_keys:
                all_keys.add(v)
                keys.append(v)
        else:
            available_keys[v].add(p)
            if p not in all_keys:
                all_keys.add(p)
                keys.append(p)
    matching_subset = all_keys
    for subset in available_keys.values():
        matching_subset &= subset

    keys = list(filter(lambda x: x in matching_subset, keys))

entries = {}

if args.sections or args.section:
    data = {
        "adva": defaultdict(dict),
        "eval": defaultdict(dict),
        "upda": defaultdict(dict),
        "etc": defaultdict(dict),
    }
else:
    data = {
        "etc": defaultdict(dict),
    }

error = [
    defaultdict(dict),
    defaultdict(dict),
]


for (variant, problem), value in variants.items():
    value.average()
    value.total()

    if args.variant:
        entries[f"{variant}"] = None
        key = problem
        v = variant
    elif args.problem:
        entries[f"{problem}"] = None
        key = variant
        v = problem
    else:
        entries[f"{variant}\n{problem}"] = None
        key = f"{variant}\n{problem}"
        v = key

    if args.sections or args.section:
        data["adva"][key][v] = (value.adva)
        data["eval"][key][v] = (value.eval)
        data["upda"][key][v] = (value.upda)
        data["etc"][key][v] = (value.etc)
    elif args.score:
        data["etc"][key][v] = (value.rounds * 1000 / value.optr)
    else:
        data["etc"][key][v] = (value.optr)

    error[0][key][v] = (value.optr - value.minmax[0])
    error[1][key][v] = (value.minmax[1] - value.optr)

width = args.width
x = np.arange(len(keys))

fig, ax = plt.subplots()
ax: matplotlib.axes.Axes # IDE Type annotation
bottoms = {key: np.zeros(len(keys)) for key in data["etc"]}
ax.ticklabel_format(axis="y",style="sci", scilimits=(0, 0), useMathText=True)

for category, values in ((k, v) for k, v in data.items() if not args.section or k in args.section):
    for index, kw in enumerate(values.items()):
        offset = index * width
        (key, weights) = kw
        weights = [weights[k] for k in keys]
        errVal = None
        if args.errors and category == "etc":
            errVal = (
                [error[0][key][k] for k in keys],
                [error[1][key][k] for k in keys],
            )

        p = ax.bar(x + offset, weights, width, label=category if args.sections or args.section else key, yerr=errVal, bottom=bottoms[key])
        bottoms[key] += weights

ax.set_title(args.title)
ax.set_ylabel("execution time (seconds)")
ax.set_xlabel("problem size")
#ax.yaxis.set_major_formatter("{x:.0e}")
ax.legend(loc="upper left")
ax.set_xticks(x + width * 0.5 * (len(data["etc"]) - 1), labels=[re.match(r"ESC63s(\d+)\.sop", k).group(1) for k in keys])
if args.log:
    ax.semilogy()

if args.output is not None:
    fig.savefig(args.output, dpi=200)
else:
    plt.show()
