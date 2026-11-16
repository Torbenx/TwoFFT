#!/usr/bin/env python3
"""Prints benchmark output files (bench/bench.c format) as one table per file:
rows = log2 size, columns = libraries, cells = ns per transform (or ratio to the first library with -r)."""
import sys

def main():
    args = sys.argv[1:]
    ratio = "-r" in args
    files = [a for a in args if a != "-r"]
    for path in files:
        data, libs = {}, []
        for line in open(path):
            parts = line.split()
            if len(parts) != 4 or parts[0] == "lib":
                continue
            lib, n, ns = parts[0], int(parts[1]), float(parts[2])
            if lib not in libs:
                libs.append(lib)
            data.setdefault(n, {})[lib] = ns
        print(f"== {path}")
        print("log2n " + " ".join(f"{l:>11}" for l in libs))
        for n in sorted(data):
            row = data[n]
            base = row.get(libs[0])
            cells = []
            for l in libs:
                if l not in row:
                    cells.append(f"{'-':>11}")
                elif ratio and l != libs[0] and base:
                    cells.append(f"{row[l] / base:>10.2f}x")
                else:
                    cells.append(f"{row[l]:>11.1f}")
            print(f"{n:>5} " + " ".join(cells))

if __name__ == "__main__":
    main()
