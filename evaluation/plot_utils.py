
from pathlib import Path
import csv
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

type Measurement = Literal["optr", "adva", "upda", "eval"]

def load_data(paths: Path | str | list[Path] | list[str], *, measurement: Measurement = "optr", variants=None, problems=None) -> defaultdict[tuple[str, str], list[float]]:
	if not isinstance(paths, list):
		paths = [paths]
	result = defaultdict(list)

	for path in paths:
		with open(path) as file:
			reader = csv.DictReader(file, delimiter=";")

			for line in reader:
				variant = line["variant"]
				problem = line["problem"]
				time = float(line[measurement])

				allow_variant = variants is None or variant in variants
				allow_problem = problems is None or problem in problems
				if allow_variant and allow_problem:
					result[(variant, problem)].append(time)
	return result
