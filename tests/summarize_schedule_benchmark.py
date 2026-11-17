"""Summarize schedule_benchmark.csv by circuit family, to check whether each strategy helps consistently.

Usage: python3 summarize_schedule_benchmark.py [schedule_benchmark.csv]

Schedules from the baseline are often invalid (the zones cannot hold every operation of a layer at once, so it moves
ions away before their operations happen), and the makespan of an invalid schedule is too short, so the strategies are
compared with the rounds strategy: the baseline with layers split into rounds which fit in the zones, which is always
valid. Comparisons with the baseline itself are only made where its schedules are valid. The lookahead strategies are
also run looking further ahead (e.g. lookahead-3 keeps an ion in its zone for an operation up to 3 layers later), and
these are compared with rounds and with looking one layer ahead. Strategies with commuting (diagonal) operations moved
into fewer layers (e.g. rounds-commute) are compared with the same strategy without.
"""

import csv
import statistics
import sys
from collections import defaultdict

STRATEGIES = ["baseline", "fast", "rounds", "lookahead", "layer-assign"]
DEPTHS = [1, 2, 3, 6]
VARIANTS = [("rounds-commute", "rounds"), ("layer-assign-commute", "layer-assign"), ("layer-assign-2-commute", "layer-assign-2")]


def variant(strategy, depth):
    """Name of a lookahead strategy looking depth layers ahead."""
    return strategy if depth == 1 else f"{strategy}-{depth}"


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

    depth_section(cases, families)
    variant_section(cases, families)


def variant_section(cases, families):
    """Compare each strategy run with commuting operations moved into fewer layers with the same strategy without."""
    present = set().union(*(r.keys() for r in cases.values()))
    pairs = [(v, b) for v, b in VARIANTS if v in present and b in present]
    if not pairs:
        return

    rows = list(cases.values())
    print("\nValid schedules with commuting operations moved (of all circuits and layouts)")
    print("  ".join(f"{v}: {sum(r[v]['valid'] for r in rows)} / {len(rows)}" for v, _ in pairs))

    for metric, title in [("makespan_us", "Makespan"), ("distance_um", "Transport distance")]:
        print(f"\n{title} reduction with commuting operations moved: median [min, max], and cases better/same/worse")
        print(f"{'family':12s}  " + "  ".join(f"{v + ' vs ' + b:>46s}" for v, b in pairs))
        for name in families + ["all"]:
            group = [r for (c, _), r in cases.items() if name in ("all", family(c))]
            cells = []
            for v, b in pairs:
                values = [reduction(r[v][metric], r[b][metric]) for r in group]
                counts = f"{sum(x > 1E-9 for x in values)}/{sum(abs(x) <= 1E-9 for x in values)}/{sum(x < -1E-9 for x in values)}"
                cells.append(f"{spread(values)} {counts:>8s}")
            print(f"{name:12s}  " + "  ".join(f"{c:>46s}" for c in cells))

    print("\nRounds with commuting operations moved: median rounds without -> with")
    print(f"{'family':12s}  " + "  ".join(f"{v:>24s}" for v, _ in pairs))
    for name in families + ["all"]:
        group = [r for (c, _), r in cases.items() if name in ("all", family(c))]
        cells = [f"{statistics.median(int(r[b]['rounds']) for r in group):g} -> {statistics.median(int(r[v]['rounds']) for r in group):g}" for v, b in pairs]
        print(f"{name:12s}  " + "  ".join(f"{c:>24s}" for c in cells))


def depth_section(cases, families):
    """Compare looking further ahead with rounds and with looking one layer ahead, where the CSV has those strategies."""
    present = set().union(*(r.keys() for r in cases.values()))
    depths = [d for d in DEPTHS if all(variant(s, d) in present for s in ("lookahead", "layer-assign"))]
    if len(depths) < 2:
        return

    print("\nValid schedules looking further ahead (of all circuits and layouts)")
    names = [variant(s, d) for s in ("lookahead", "layer-assign") for d in depths[1:]]
    rows = list(cases.values())
    print("  ".join(f"{n}: {sum(r[n]['valid'] for r in rows)} / {len(rows)}" for n in names))

    for metric, title in [("makespan_us", "Makespan"), ("distance_um", "Transport distance")]:
        for strategy in ("lookahead", "layer-assign"):
            print(f"\n{title} reduction of {strategy} by lookahead depth: median vs rounds, and cases better/same/worse "
                  f"than looking 1 layer ahead")
            print(f"{'family':12s}  " + "  ".join(f"{'depth ' + str(d):>22s}" for d in depths))
            for name in families + ["all"]:
                group = [r for (c, _), r in cases.items() if name in ("all", family(c))]
                cells = []
                for d in depths:
                    vs_rounds = [reduction(r[variant(strategy, d)][metric], r["rounds"][metric]) for r in group]
                    vs_one = [reduction(r[variant(strategy, d)][metric], r[strategy][metric]) for r in group]
                    counts = f"{sum(v > 1E-9 for v in vs_one)}/{sum(abs(v) <= 1E-9 for v in vs_one)}/{sum(v < -1E-9 for v in vs_one)}"
                    cells.append(f"{statistics.median(vs_rounds):5.1f}% {counts:>9s}" if d > 1 else f"{statistics.median(vs_rounds):5.1f}%")
                print(f"{name:12s}  " + "  ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
