"""Summarize schedule_benchmark.csv by circuit family, to check whether each strategy helps consistently.

Usage: python3 summarize_schedule_benchmark.py [schedule_benchmark.csv]

Schedules from the baseline are often invalid (the zones cannot hold every operation of a layer at once, so it moves
ions away before their operations happen), and the makespan of an invalid schedule is too short, so the strategies are
compared with the rounds strategy: the baseline with layers split into rounds which fit in the zones, which is always
valid. Comparisons with the baseline itself are only made where its schedules are valid.
"""

import csv
import statistics
import sys
from collections import defaultdict

STRATEGIES = ["baseline", "fast", "rounds", "lookahead", "layer-assign"]


def load(filename):
    cases = defaultdict(dict)
    with open(filename) as f:
        for row in csv.DictReader(f):
            row["valid"] = row["scheduled"] == "1" and row["invalid"] == "0"
            for key in ("makespan_us", "distance_um", "run_time_us"):
                row[key] = float(row[key])
            cases[(row["circuit"], row["layout"])][row["strategy"]] = row
    return cases


def family(circuit):
    return circuit.rsplit("_", 1)[0]


def reduction(new, old):
    return 100.0 * (1.0 - new / old) if old > 0 else 0.0


def spread(values):
    if not values:
        return "-"
    return f"{statistics.median(values):5.1f}% [{min(values):5.1f}, {max(values):5.1f}]"


def main():
    cases = load(sys.argv[1] if len(sys.argv) > 1 else "schedule_benchmark.csv")
    families = sorted({family(circuit) for circuit, _ in cases})

    print("Valid schedules (of all circuits and layouts)")
    print(f"{'family':12s} " + " ".join(f"{s:>13s}" for s in STRATEGIES))
    for name in families + ["all"]:
        rows = [r for (c, _), r in cases.items() if name in ("all", family(c))]
        print(f"{name:12s} " + " ".join(f"{sum(r[s]['valid'] for r in rows):>6d} / {len(rows):<4d}" for s in STRATEGIES))

    comparisons = [
        ("lookahead", "rounds"),
        ("layer-assign", "rounds"),
        ("layer-assign", "lookahead"),
    ]
    header = [f"{new} vs {old}" for new, old in comparisons]

    for metric, title in [("makespan_us", "Makespan"), ("distance_um", "Transport distance")]:
        print(f"\n{title} reduction, median [min, max] over every circuit and layout (negative is worse)")
        print(f"{'family':12s}  " + "  ".join(f"{h:>27s}" for h in header) + f"  {'cases better/same/worse':>24s}")
        for name in families + ["all"]:
            rows = [r for (c, _), r in cases.items() if name in ("all", family(c))]
            columns = [[reduction(r[new][metric], r[old][metric]) for r in rows] for new, old in comparisons]
            best = columns[1]
            counts = f"{sum(v > 1E-9 for v in best)}/{sum(abs(v) <= 1E-9 for v in best)}/{sum(v < -1E-9 for v in best)}"
            print(f"{name:12s}  " + "  ".join(f"{spread(c):>27s}" for c in columns) + f"  {counts:>24s}")

    print("\nMakespan reduction of layer-assign vs baseline, where the baseline is valid")
    for name in families + ["all"]:
        rows = [r for (c, _), r in cases.items() if name in ("all", family(c)) and r["baseline"]["valid"]]
        values = [reduction(r["layer-assign"]["makespan_us"], r["baseline"]["makespan_us"]) for r in rows]
        print(f"{name:12s} {len(rows):>3d} cases  {spread(values)}")

    print("\nScheduler run time (us), median [min, max]")
    print(f"{'family':12s} " + " ".join(f"{s:>24s}" for s in STRATEGIES) + f" {'fast speedup':>14s}")
    for name in families + ["all"]:
        rows = [r for (c, _), r in cases.items() if name in ("all", family(c))]
        times = {s: [r[s]["run_time_us"] for r in rows] for s in STRATEGIES}
        speedups = [r["baseline"]["run_time_us"] / r["fast"]["run_time_us"] for r in rows]
        print(f"{name:12s} " + " ".join(f"{statistics.median(t):8.2f} [{min(t):5.1f}, {max(t):6.1f}]" for t in times.values())
              + f" {statistics.median(speedups):13.2f}x")


if __name__ == "__main__":
    main()
