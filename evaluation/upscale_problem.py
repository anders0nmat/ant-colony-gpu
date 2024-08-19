import numpy as np
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("file")
g1 = parser.add_mutually_exclusive_group(required=True)
g1.add_argument("-t", "--times", type=int)
g1.add_argument("-s", "--size", type=int)
parser.add_argument("--output", "-o")
args = parser.parse_args()

def ceildiv(a: int, b: int) -> int:
    return -(a // -b)

def scale_matrix(weights: np.ndarray, repeat: int) -> np.ndarray:
    new_matrix = np.tile(weights, (repeat, repeat))
    new_matrix = np.pad(new_matrix, 1, constant_values=((0, -1), (-1, 0)))
    new_matrix[0, 0] = 0
    new_matrix[0, -1] = 1000000
    return new_matrix

def size_matrix(weights: np.ndarray, new_size: int) -> np.ndarray:
    current_size = weights.shape[0]
    scale = ceildiv(new_size, current_size) if new_size > current_size else 1
    new_matrix = np.tile(weights, (scale, scale))
    new_matrix = new_matrix[:new_size, :new_size]
    new_matrix = np.pad(new_matrix, 1, constant_values=((0, -1), (-1, 0)))
    new_matrix[0, 0] = 0
    new_matrix[0, -1] = 1000000
    return new_matrix


def has_dependency_loop(arr: np.ndarray) -> bool:
    arr = np.clip(arr * -1, 0, 1) == 0
    size = arr.shape[0]
    def first_valid():
        for i in range(arr.shape[0]):
            if np.all(arr[i]):
                return i
        return None

    for _ in range(size):
        nextNode = first_valid()
        if nextNode is None:
            return True
        arr = np.delete(arr, nextNode, axis=0)
        arr = np.delete(arr, nextNode, axis=1)
    return False


file_text = Path(args.file).read_text()

[headers, matrix] = [part.strip() for part in file_text.split("EDGE_WEIGHT_SECTION", maxsplit=1)]

headers = (line.partition(":") for line in headers.splitlines())
headers = {key.strip(): value.strip() for (key, _, value) in headers}

matrix = matrix.split("EOF", maxsplit=1)[0].strip()
matrix = [line for line in matrix.splitlines() if line]
dim, matrix = int(matrix[0]), matrix[1:]
matrix = [np.fromstring(line, sep=' ', dtype=int, count=dim) for line in matrix]
matrix = np.array(matrix)

(width, height) = matrix.shape
assert width == height 

weights = matrix[1:-1, 1:-1]

if args.times:
    new_matrix = scale_matrix(weights, repeat=args.times)
    headers["NAME"] = headers["NAME"].replace(".sop", f"x{args.times}.sop")
elif args.size:
    new_matrix = size_matrix(weights, new_size=args.size - 2)
    headers["NAME"] = headers["NAME"].replace(".sop", f"s{args.size}.sop")
else:
    raise ValueError("Neither --times nor --size was specified")

if has_dependency_loop(new_matrix):
    raise ValueError("Generated Graph has dependency loop")

headers["DIMENSION"] = str(new_matrix.shape[0])
headers["SOLUTION_BOUNDS"] = "-1"

out_file = [f"{key}: {value}" for (key, value) in headers.items()]
out_file += [
    "EDGE_WEIGHT_SECTION",
    str(new_matrix.shape[0]),
    #np.array_str(new_matrix, max_line_width=10_000_000_000_000).replace("[", "").replace("]", ""),
    #"EOF",
]

#Path(headers["NAME"]).write_text("\n".join(out_file))

np.savetxt(
    args.output if args.output else headers["NAME"],
    new_matrix,
    fmt="%d",
    header="\n".join(out_file),
    footer="EOF",
    comments='')
