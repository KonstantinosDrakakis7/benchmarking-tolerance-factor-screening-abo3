"""
03_topology.py -- structural topology labels and duplicate detection.

For every entry: B-O bonds from CrystalNN; for each pair of BO6 polyhedra the number
of shared O atoms (1 corner, 2 edge, >= 3 face), periodic images tracked explicitly;
A-site environment from the number of B cations within 1.25 (or 1.30) x the
shortest A-B distance and of O within 1.25 x the shortest A-O distance.

Labels (first that applies):
  non-perovskite (B not octahedral)
  non-perovskite (face-sharing BO6)
  non-perovskite (edge-sharing BO6)
  non-perovskite (corner-sharing, not 3D)
  perovskite        3D corner-sharing BO6 and every A in a cavity of 8 B cations
                    (1.25 x; or 1.30 x if A also has >= 8 O within 1.25 x)
  non-perovskite (corner-sharing, A outside cavity)   e.g. LiNbO3-, pyrochlore-type
  ambiguous         conflicting criteria (excluded from the statistics)
Duplicates: pymatgen StructureMatcher (default tolerances) per composition.

Run:  python 03_topology.py --check   (reference compounds only)
      python 03_topology.py           (all entries)
Output: ABO3_topology_entries.csv, ABO3_topology_compositions.csv
"""

import sys
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher

IN_CSV = "ABO3_radii_t_mu.csv"
IN_STRUCT = "MP_ABO3_structures.json"
OUT_ENTRIES = "ABO3_topology_entries.csv"
OUT_COMPS = "ABO3_topology_compositions.csv"

A_CN_MIN = 8
A_CUTOFF_FACTOR = 1.25
CHECK_FORMULAS = {   # expected label of the ground state, for a sanity check
    "SrTiO3": "perovskite", "KNbO3": "perovskite", "CaTiO3": "perovskite",
    "TiFeO3": "edge", "MgTiO3": "edge", "BaNiO3": "face",
    "LiNbO3": "corner-sharing, A outside cavity", "RbNbO3": "see all 3 entries",
    "NaNbO3": "see all entries",
}

cnn = CrystalNN()


def _sym(site):
    return site.specie.symbol


def octahedral_connectivity(s: Structure, B: str):
    b_idx = [i for i, site in enumerate(s) if _sym(site) == B]
    b_O = {}                         # B index -> [(O index, image)]
    o_to_b = defaultdict(list)       # O index -> [(B index, image of B relative to O)]
    for i in b_idx:
        nbrs = [(n["site_index"], tuple(int(round(x)) for x in n["image"]))
                for n in cnn.get_nn_info(s, i) if _sym(n["site"]) == "O"]
        b_O[i] = nbrs
        for o, img in nbrs:
            o_to_b[o].append((i, tuple(-x for x in img)))

    per_B = []
    for i in b_idx:
        shared = Counter()
        for o, img in b_O[i]:
            for j, rel in o_to_b[o]:
                tot = tuple(a + b for a, b in zip(img, rel))
                if (j, tot) != (i, (0, 0, 0)):
                    shared[(j, tot)] += 1
        per_B.append({
            "cn": len(b_O[i]),
            "corner": sum(v == 1 for v in shared.values()),
            "edge": sum(v == 2 for v in shared.values()),
            "face": sum(v >= 3 for v in shared.values()),
        })
    return per_B


def a_site_cn(s: Structure, A: str):
    cut_counts, cnn_counts = [], []
    for i, site in enumerate(s):
        if _sym(site) != A:
            continue
        d = sorted(n.nn_distance for n in s.get_neighbors(site, 4.0) if _sym(n) == "O")
        cut_counts.append(sum(x <= A_CUTOFF_FACTOR * d[0] for x in d) if d else 0)
        cnn_counts.append(sum(_sym(n["site"]) == "O" for n in cnn.get_nn_info(s, i)))
    return min(cut_counts), min(cnn_counts)


def a_site_cage(s: Structure, A: str, B: str, factor: float):
    """Min over A sites of the number of B within factor x shortest A-B distance."""
    counts = []
    for site in s:
        if _sym(site) != A:
            continue
        d = sorted(n.nn_distance for n in s.get_neighbors(site, 6.5) if _sym(n) == B)
        counts.append(sum(x <= factor * d[0] + 1e-6 for x in d) if d else 0)
    return min(counts)


def classify(s: Structure, A: str, B: str):
    per_B = octahedral_connectivity(s, B)
    a_cut, a_cnn = a_site_cn(s, A)
    cage125, cage130 = a_site_cage(s, A, B, 1.25), a_site_cage(s, A, B, 1.30)
    info = {
        "B_CN_min": min(p["cn"] for p in per_B), "B_CN_max": max(p["cn"] for p in per_B),
        "BO6_corner_min": min(p["corner"] for p in per_B),
        "BO6_edge_max": max(p["edge"] for p in per_B),
        "BO6_face_max": max(p["face"] for p in per_B),
        "A_CN_cutoff": a_cut, "A_CN_crystalnn": a_cnn,
        "A_cage_B_1.25": cage125, "A_cage_B_1.30": cage130,
    }
    if any(p["cn"] != 6 for p in per_B):
        label = "non-perovskite (B not octahedral)"
    elif info["BO6_face_max"] > 0:
        label = "non-perovskite (face-sharing BO6)"
    elif info["BO6_edge_max"] > 0:
        label = "non-perovskite (edge-sharing BO6)"
    elif any(p["corner"] != 6 for p in per_B):
        label = "non-perovskite (corner-sharing, not 3D)"
    elif cage125 >= 8 or (cage130 >= 8 and a_cut >= A_CN_MIN):
        label = "perovskite"
    elif cage125 <= 7 and a_cut <= 7:
        label = "non-perovskite (corner-sharing, A outside cavity)"
    else:
        label = "ambiguous"
    info["topology"] = label
    return info


def main(check_only=False):
    df = pd.read_csv(IN_CSV, sep=";")
    df = df[df["exclusion_reason"].fillna("") == ""].copy()
    if check_only:
        df = df[df["Formula"].isin(CHECK_FORMULAS)]
    with open(IN_STRUCT) as fh:
        structs = json.load(fh)

    rows = []
    for k, r in enumerate(df.itertuples(index=False), 1):
        mid = r[df.columns.get_loc("Material ID")]
        try:
            s = Structure.from_dict(structs[mid])
            info = classify(s, r[df.columns.get_loc("Atom A")], r[df.columns.get_loc("Atom B")])
        except Exception as exc:
            info = {"topology": f"error: {exc}"}
        info["Material ID"] = mid
        rows.append(info)
        if k % 100 == 0:
            print(f"  {k}/{len(df)}")
    ent = df.merge(pd.DataFrame(rows), on="Material ID")

    # Duplicate detection per composition
    sm = StructureMatcher()
    ent["structure_group"] = -1
    for f, sub in ent.groupby("Formula"):
        objs = {m: Structure.from_dict(structs[m]) for m in sub["Material ID"]}
        by_id = {id(st): m for m, st in objs.items()}
        for g, members in enumerate(sm.group_structures(list(objs.values()))):
            for st in members:
                ent.loc[ent["Material ID"] == by_id[id(st)], "structure_group"] = g
    ent.to_csv(OUT_ENTRIES, index=False, sep=";")

    if check_only:
        cols = ["Formula", "Material ID", "Space Group", "Energy Above Hull", "topology",
                "B_CN_min", "BO6_edge_max", "BO6_face_max", "A_CN_cutoff", "A_cage_B_1.25"]
        print(ent.sort_values(["Formula", "Energy Above Hull"])[cols].to_string(index=False))
        print("\nExpected (ground states):", json.dumps(CHECK_FORMULAS, indent=1))
        return

    # Per-composition summary
    out = []
    for f, sub in ent.groupby("Formula"):
        sub = sub.sort_values("Energy Above Hull")
        per = sub[sub.topology == "perovskite"]
        non = sub[sub.topology.str.startswith("non-perovskite")]
        e_p = per["Energy Above Hull"].min() if len(per) else np.nan
        e_n = non["Energy Above Hull"].min() if len(non) else np.nan
        out.append({
            "Formula": f, "admissible": bool(sub["admissible"].iloc[0]),
            "B_lone_pair": bool(sub["B_lone_pair"].iloc[0]),
            "n_entries": len(sub), "n_distinct_structures": sub["structure_group"].nunique(),
            "ground_state_topology": sub["topology"].iloc[0],
            "lowest_Ehull_meV": 1000 * sub["Energy Above Hull"].iloc[0],
            "lowest_perovskite_meV": 1000 * e_p, "lowest_nonperovskite_meV": 1000 * e_n,
            "gap_nonperov_minus_perov_meV": 1000 * (e_n - e_p),
            "any_icsd_entry": bool((~sub["Theoretical"].astype(bool)).any()),
        })
    comps = pd.DataFrame(out)
    comps.to_csv(OUT_COMPS, index=False, sep=";")

    print("\nGround-state topology, admissible vs non-admissible (baseline):")
    print(pd.crosstab(comps["ground_state_topology"], comps["admissible"], margins=True))
    both = comps.dropna(subset=["gap_nonperov_minus_perov_meV"])
    print(f"\nCompositions with both a perovskite and a non-perovskite entry: {len(both)}")
    print(both.groupby("admissible")["gap_nonperov_minus_perov_meV"].describe())
    print(f"\nSaved -> {OUT_ENTRIES}, {OUT_COMPS}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
