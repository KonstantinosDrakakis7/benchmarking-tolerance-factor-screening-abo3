"""
05_analysis.py -- statistics of the manuscript (pandas only, no API, no pymatgen).

Inputs : ABO3_topology_struct.csv (04), ABO3_descriptors_bartel.csv (04),
         TableS1.csv (Bartel et al. 2019, downloaded by 04), MP_ABO3_raw.csv (01)
Outputs: ABO3_labels_and_descriptors.csv (one row per labelled composition) and a
         console report containing every number in the paper.

Choices (all fixed before computing descriptor accuracies, except where stated):
  * entries more than 0.5 eV/atom above the hull are discarded (corrupted or far
    from stable); sensitivity without this filter is reported;
  * an entry labelled "perovskite" by 06 whose A cation has fewer than 4 O within
    1.25 x its shortest A-O distance (triangular or pyramidal oxoanion-forming cations such
    as B, C, N, Se, Te, I) is relabelled "non-perovskite (A under-coordinated)".
    This rule was added AFTER the comparison with Bartel's labels revealed
    calcite-type borates and carbonates labelled as perovskites; results without it
    are reported;
  * label L1 = topology of the lowest-energy ICSD-matched entry;
    label L2 = perovskite if ANY ICSD-matched entry is a perovskite;
  * descriptors t, mu, tau from the reference implementation of Bartel et al.;
    rules: conventional window (0.80 <= t <= 1.10 and 0.414 <= mu <= 0.732),
    Bartel's t bounds (0.825 < t < 1.059), tau < 4.18;
  * composition sets: all; without oxoanion-forming elements (H, B, C, N, Si, P, S,
    Cl, Ge, As, Se, Br, Te, I on either site); compositions absent from Bartel's
    experimental set ("out-of-sample"), with and without oxoanion-forming elements.
"""

import re
from math import sqrt, comb

import numpy as np
import pandas as pd

E_MAX = 0.5
A_O_MIN = 4
OXY = {"H", "B", "C", "N", "Si", "P", "S", "Cl", "Ge", "As", "Se", "Br", "Te", "I"}
BINS = [0, 0.825, 0.90, 1.00, 1.059, 1.10 + 1e-9, 9]


def key(formula):
    """Order-independent composition key (element, count) for matching formula strings."""
    els = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    return tuple(sorted((e, int(n or 1)) for e, n in els))


def cations(formula):
    return set(re.findall(r"[A-Z][a-z]?", formula)) - {"O"}


def wilson(k, n, z=1.96):
    if n == 0:
        return "n/a"
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * p:.1f}% [{100 * (c - h):.1f}-{100 * (c + h):.1f}]"


def stats(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    n = tp + fp + fn + tn
    return dict(n=n, acc=wilson(tp + tn, n), rec=100 * tp / max(tp + fn, 1),
                spec=100 * tn / max(tn + fp, 1), tp=tp, fp=fp, fn=fn, tn=tn)


def labels(st, a_rule=True, e_filter=True):
    st = st[(st.topology_struct != "ambiguous") & ~st.topology_struct.str.startswith("error")].copy()
    if e_filter:
        st = st[st["Energy Above Hull"] <= E_MAX]
    if a_rule:
        m = (st.topology_struct == "perovskite") & (st.A_CN_cutoff_struct < A_O_MIN)
        st.loc[m, "topology_struct"] = "non-perovskite (A under-coordinated)"
    icsd = st[st.Theoretical.astype(str).str.lower() == "false"].sort_values("Energy Above Hull")
    g = icsd.groupby("Formula")
    out = pd.DataFrame({"L1_topology": g.topology_struct.first(),
                        "L2": g.topology_struct.apply(lambda s: (s == "perovskite").any())})
    out["L1"] = out.L1_topology == "perovskite"
    return out, st


def main():
    raw = pd.read_csv("MP_ABO3_raw.csv", sep=";")
    st0 = pd.read_csv("ABO3_topology_struct.csv", sep=";")
    bd = pd.read_csv("ABO3_descriptors_bartel.csv", sep=";").set_index("Formula")
    s1 = pd.read_csv("TableS1.csv")
    ox = s1[s1.ABX3.str.endswith("O3")].copy()
    ox["key"] = ox.ABX3.map(key)
    exp = dict(zip(ox.key, ox.exp_label))

    print("== Data and labels")
    print(f"entries {len(raw)}, compositions {raw.Formula.nunique()}")
    print(f"bond-valence failures {int((~st0.bv_ok.fillna(False).astype(bool)).sum())}; "
          f"site rules {st0.site_rule.value_counts().to_dict()}")
    print(f"ambiguous {int((st0.topology_struct == 'ambiguous').sum())}, "
          f"errors {int(st0.topology_struct.str.startswith('error').sum())}, "
          f"entries > {E_MAX} eV/atom {int((st0['Energy Above Hull'] > E_MAX).sum())}")
    lab, st = labels(st0)
    n_relab = int((st.topology_struct == "non-perovskite (A under-coordinated)").sum())
    print(f"entries used {len(st)}; relabelled by A-O rule {n_relab} "
          f"(A elements {st[st.topology_struct.str.contains('under')].A_struct.value_counts().to_dict()})")
    print(f"entry topology: {st.topology_struct.value_counts().to_dict()}")
    print(f"compositions with ICSD-matched entry {len(lab)}; L1 perovskite {int(lab.L1.sum())}, "
          f"L2 perovskite {int(lab.L2.sum())}, L1 != L2: {int((lab.L1 != lab.L2).sum())}")

    lab["key"] = [key(f) for f in lab.index]
    lab["exp_label"] = lab.key.map(exp)
    lab["in_bartel"] = lab.exp_label.notna()
    lab["oxy"] = [bool(cations(f) & OXY) for f in lab.index]
    lab = lab.join(bd[["bartel_t", "bartel_mu", "bartel_tau", "bartel_nA", "bartel_nB"]])

    print(f"\n== Agreement with Bartel's experimental labels ({len(lab[lab.in_bartel])} shared oxide compositions)")
    o = lab[lab.in_bartel]
    e1 = o.exp_label == 1
    for L in ["L1", "L2"]:
        print(f"  {L}: {wilson(int((o[L] == e1).sum()), len(o))}  "
              f"(exp perovskite & ours not: {int((e1 & ~o[L]).sum())}; ours perovskite & exp not: {int((~e1 & o[L]).sum())})")
    for name, kw in [("no A-O rule", dict(a_rule=False)), ("no energy filter", dict(e_filter=False))]:
        l2, _ = labels(st0, **kw)
        l2 = l2[[key(f) in exp for f in l2.index]]
        ee = pd.Series([exp[key(f)] == 1 for f in l2.index], index=l2.index)
        print(f"  sensitivity, {name}: L1 {wilson(int((l2.L1 == ee).sum()), len(l2))}, "
              f"L2 {wilson(int((l2.L2 == ee).sum()), len(l2))}")
    dis = o[(o.L1 != e1) | (o.L2 != e1)]
    print("  disagreements (exp label, L1 topology, L2):")
    print(dis[["exp_label", "L1_topology", "L2"]].to_string())

    lab.to_csv("ABO3_labels_and_descriptors.csv", sep=";")
    d = lab.dropna(subset=["bartel_t", "bartel_mu", "bartel_tau"]).copy()
    d["conv"] = d.bartel_t.between(0.80, 1.10) & d.bartel_mu.between(0.414, 0.732)
    d["tb"] = (d.bartel_t > 0.825) & (d.bartel_t < 1.059)
    d["tau"] = d.bartel_tau < 4.18
    print(f"\n== Descriptor accuracy (compositions with Bartel descriptors: {len(d)} of {len(lab)})")
    sets = {"all": d, "oxoanion only": d[d.oxy], "no oxoanion": d[~d.oxy], "out-of-sample": d[~d.in_bartel],
            "out-of-sample, no oxoanion": d[~d.in_bartel & ~d.oxy]}
    for sname, s in sets.items():
        for L in ["L1", "L2"]:
            tr = s[L]
            base = max(int(tr.sum()), len(tr) - int(tr.sum()))
            row = f"  {sname:27s} {L} n={len(s):3d} baseline {wilson(base, len(s))}"
            for col, nm in [("conv", "conv"), ("tb", "t-bounds"), ("tau", "tau")]:
                r = stats(s[col], tr)
                row += f" | {nm} {r['acc']} (rec {r['rec']:.0f}, spec {r['spec']:.0f})"
            print(row)

    print("\n== Failure decomposition (all compositions, L1)")
    for col, nm in [("conv", "conventional window"), ("tau", "tau")]:
        fp = d[d[col] & ~d.L1]
        fn = d[~d[col] & d.L1]
        print(f"  {nm}: FP {len(fp)} by L1 topology {fp.L1_topology.value_counts().to_dict()}")
        if col == "conv":
            print(f"     FN {len(fn)}: mu<0.414 {(fn.bartel_mu < 0.414).sum()}, t<0.80 {(fn.bartel_t < 0.80).sum()}, "
                  f"t>1.10 {(fn.bartel_t > 1.10).sum()}, mu>0.732 {(fn.bartel_mu > 0.732).sum()}")
        else:
            print(f"     FN {len(fn)}")

    print("\n== Perovskite fraction by Bartel t (mu inside window, all compositions)")
    z = d[d.bartel_mu.between(0.414, 0.732)].copy()
    z["bin"] = pd.cut(z.bartel_t, BINS, right=False)
    for b, q in z.groupby("bin", observed=True):
        print(f"  {str(b):16s} L1 {int(q.L1.sum())}/{len(q)}   L2 {int(q.L2.sum())}/{len(q)}")
    up = z[(z.bartel_t >= 1.059) & (z.bartel_t <= 1.10)].sort_values("bartel_t")
    print("  upper window 1.059 <= t <= 1.10:")
    print(up[["bartel_t", "bartel_tau", "L1_topology", "L2", "exp_label"]].round(3).to_string())

    print("\n== Case-study entries")
    for f in ["RbNbO3", "RbTaO3", "CsNbO3", "KSbO3"]:
        q = st[st.Formula == f].sort_values("Energy Above Hull")
        print(q[["Formula", "Material ID", "Space Group", "Energy Above Hull", "Volume per atom",
                 "Theoretical", "B_struct", "B_bv", "topology_struct"]].round(4).to_string(index=False))
        if f in d.index:
            print(f"   Bartel descriptors: t={d.loc[f, 'bartel_t']:.3f} mu={d.loc[f, 'bartel_mu']:.3f} "
                  f"tau={d.loc[f, 'bartel_tau']:.3f}; Bartel experimental label {d.loc[f, 'exp_label']}")


if __name__ == "__main__":
    main()
