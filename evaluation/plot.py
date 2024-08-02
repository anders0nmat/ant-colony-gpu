import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from dataclasses import dataclass

argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("file", help="CSV-file with result data")
argumentParser.add_argument("-w", "--width", type=float, default=0.5, help="Width of bars")
argumentParser.add_argument("-e", "--errors", action="store_true", help="Show error bars")

g1 = argumentParser.add_mutually_exclusive_group()
g1.add_argument("-v", "--variant", action="store_true", help="Use variant name as caption only")
g1.add_argument("-p", "--problem", action="store_true", help="Use problem name as caption only")

g2 = argumentParser.add_mutually_exclusive_group()
g2.add_argument("-s", "--sections", action="store_true", help="Show execution sections: adva, eval, upda")
g2.add_argument("-c", "--score", action="store_true", help="Show normalized RPS score instead of execution time")

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

with open(args.file) as f:
    data = csv.DictReader(f, delimiter=";")
    
    for line in data:
        key = (line["variant"], line["problem"])

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

entries = []

if args.sections:
    data = {
        "adva": [],
        "eval": [],
        "upda": [],
        "etc": [],
    }
else:
    data = {
        "etc": [],
    }

error = [
    [],
    []
]


for (variant, problem), value in variants.items():
    value.average()
    value.total()
    
    if value.optr > 1000:
        continue

    if args.variant:
        entries.append(f"{variant}")
    elif args.problem:
        entries.append(f"{problem}")
    else:
        entries.append(f"{variant}\n{problem}")

    if args.sections:
        data["adva"].append(value.adva)
        data["eval"].append(value.eval)
        data["upda"].append(value.upda)
        data["etc"].append(value.etc)
    elif args.score:
        data["etc"].append(value.rounds / value.optr)
    else:
        data["etc"].append(value.optr)

    error[0].append(value.optr - value.minmax[0])
    error[1].append(value.minmax[1] - value.optr)

width = args.width

fig, ax = plt.subplots()
bottom = np.zeros(len(entries))

for boolean, weight_count in data.items():
    p = ax.bar(entries, weight_count, width, label=boolean, yerr=error if args.errors and boolean == "etc" else None, bottom=bottom)
    bottom += weight_count

if args.score:
    ax.set_title("Rounds per second of Variants on example Problems (higher is better)")
else:
    ax.set_title("Execution Time of Variants on example Problems (lower is better)")
ax.legend(loc="upper right")
#ax.semilogy()
#ax.set_yscale("logit")

plt.show()
