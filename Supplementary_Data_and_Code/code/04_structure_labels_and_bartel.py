"""
04_structure_labels_and_bartel.py -- structure-based site assignment, topology labels,
reference descriptors, and comparison with experimental labels.

Needs: pymatgen, pandas, numpy, scikit-learn, internet access (GitHub).
Inputs (same folder): MP_ABO3_raw.csv, MP_ABO3_structures.json, 02_radii_tolerance.py,
                      03_topology.py
Outputs:
  ABO3_topology_struct.csv     one row per entry: B site chosen FROM THE STRUCTURE
                               (octahedral cation), bond-valence oxidation states,
                               topology label, Shannon t/mu with these sites/valences
  ABO3_descriptors_bartel.csv  one row per composition: t, mu and tau computed with
                               the reference implementation of Bartel et al. (2019)
                               (their A/B assignment, oxidation states and radii)
  bartel_label_comparison.csv  compositions present in both MP and Bartel's 576-compound
                               experimental set (oxides), with both labels
  04_run_report.txt            counts and failures (send this file back)

Site rule (per entry, from CrystalNN O-coordination of each cation):
  1. if exactly one cation element is 6-coordinated on all its sites -> that is B;
  2. if both are -> B is the one with the higher bond-valence oxidation state,
     then the one with the shorter mean M-O distance;
  3. if neither is -> B is the one with the lower mean coordination number, then the
     shorter mean M-O distance (the entry will be labelled "B not octahedral").

Run:  python 04_structure_labels_and_bartel.py --check   (about 10 reference compositions)
      python 04_structure_labels_and_bartel.py           (all entries)
"""

import importlib.util
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from pymatgen.analysis.bond_valence import BVAnalyzer

CHECK = ["SrTiO3", "TbMnO3", "NdTiO3", "TbCoO3", "TlCrO3", "MgTiO3", "LiNbO3",
         "RbNbO3", "BaNiO3", "YMnO3", "LuFeO3", "CaTiO3"]
BARTEL_RAW = "https://raw.githubusercontent.com/CJBartel/perovskite-stability/master/"
BARTEL_FILES = ["TableS1.csv", "PredictPerovskites.py", "Shannon_radii_dict.json",
                "electronegativities.csv"]
# SHA-256 of the files used for the paper (downloaded 2 October 2026). The repository has no
# tagged release, so the content itself is pinned: a mismatch stops the script.
BARTEL_SHA256 = {
    "PredictPerovskites.py": "2a33a186b1dd75281f8310f2bbf465a73214776a791dd411d9a530bad3167446",
    "TableS1.csv": "bea471ddb1ed63dcb94cc9cbc58c73f3bc1a12918e96cf24818b3aea2cbe3656",
    "Shannon_radii_dict.json": "322870e5a1fb45e168b4ff6c8e262f88122fe5c662b1f9c5cc9676ecb74b425f",
    "electronegativities.csv": "94e26be5fd3fcd156b0c71dd8588c61947289d7e25cdeece3c277a3df00255b8",
}

REPORT = []


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    REPORT.append(msg)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


topo = load_module("topo", "03_topology.py")      # CrystalNN instance + classify()
radii = load_module("radii", "02_radii_tolerance.py")  # Shannon radius helpers


def o_environment(s, el):
    """CN by O and mean M-O distance over all sites of element el."""
    cns, dists = [], []
    for i, site in enumerate(s):
        if site.specie.symbol != el:
            continue
        nn = [n for n in topo.cnn.get_nn_info(s, i) if n["site"].specie.symbol == "O"]
        cns.append(len(nn))
        dists += [site.distance(n["site"]) for n in nn]
    return cns, (float(np.mean(dists)) if dists else np.nan)


def bv_valences(s):
    vals = BVAnalyzer().get_valences(s)
    out = {}
    for site, v in zip(s, vals):
        out.setdefault(site.specie.symbol, []).append(v)
    return {el: v for el, v in out.items()}


def assign_sites(s, cations, bv):
    env = {el: o_environment(s, el) for el in cations}
    all6 = {el: all(c == 6 for c in env[el][0]) and len(env[el][0]) > 0 for el in cations}
    c1, c2 = cations
    if all6[c1] != all6[c2]:
        B, rule = (c1 if all6[c1] else c2), "only octahedral cation"
    elif all6[c1] and all6[c2]:
        v1 = np.mean(bv[c1]) if bv else np.nan
        v2 = np.mean(bv[c2]) if bv else np.nan
        if bv and v1 != v2:
            B, rule = (c1 if v1 > v2 else c2), "both octahedral: higher valence"
        else:
            B, rule = (c1 if env[c1][1] <= env[c2][1] else c2), "both octahedral: shorter M-O"
    else:
        m1, m2 = np.mean(env[c1][0]), np.mean(env[c2][0])
        if m1 != m2:
            B, rule = (c1 if m1 < m2 else c2), "neither octahedral: lower CN"
        else:
            B, rule = (c1 if env[c1][1] <= env[c2][1] else c2), "neither octahedral: shorter M-O"
    A = c2 if B == c1 else c1
    return A, B, rule, env


def shannon_t_mu(A, a_ox, B, b_ox):
    try:
        ra, a_cn = radii.a_site_radius(A, a_ox)
        rb, _ = radii.shannon(B, b_ox, "VI")
        if ra is None or rb is None:
            return np.nan, np.nan, None
        return radii.tol(ra, rb), rb / radii.R_O, a_cn
    except Exception:
        return np.nan, np.nan, None


def structure_part(raw, structs, check):
    rows = []
    df = raw[raw.Formula.isin(CHECK)] if check else raw
    t0 = time.time()
    for k, r in enumerate(df.itertuples(index=False), 1):
        mid, f = r[0], r[1]
        row = {"Material ID": mid, "Formula": f}
        try:
            s = Structure.from_dict(structs[mid])
            cations = [el.symbol for el in s.composition.elements if el.symbol != "O"]
            try:
                bv = bv_valences(s)
                row["bv_ok"] = True
            except Exception as exc:
                bv, row["bv_ok"] = None, False
                row["bv_error"] = str(exc)[:80]
            A, B, rule, env = assign_sites(s, cations, bv)
            row.update({"A_struct": A, "B_struct": B, "site_rule": rule,
                        "A_CN_O": ";".join(map(str, env[A][0])), "B_CN_O": ";".join(map(str, env[B][0])),
                        "A_mean_MO": round(env[A][1], 3), "B_mean_MO": round(env[B][1], 3)})
            if bv:
                a_ox, b_ox = float(np.mean(bv[A])), float(np.mean(bv[B]))
                row.update({"A_bv": a_ox, "B_bv": b_ox,
                            "bv_mixed": len(set(bv[A])) > 1 or len(set(bv[B])) > 1})
                if a_ox.is_integer() and b_ox.is_integer():
                    t, mu, a_cn = shannon_t_mu(A, int(a_ox), B, int(b_ox))
                    row.update({"t_struct": t, "mu_struct": mu, "A_CN_radius": a_cn})
            info = topo.classify(s, A, B)
            row.update({f"{k2}_struct" if k2 != "topology" else "topology_struct": v
                        for k2, v in info.items()})
        except Exception as exc:
            row["topology_struct"] = f"error: {str(exc)[:80]}"
        rows.append(row)
        if k % 100 == 0:
            log(f"  {k}/{len(df)} entries, {time.time() - t0:.0f} s")
    out = df[["Material ID", "Formula", "Energy Above Hull", "Volume per atom", "Space Group",
              "Theoretical"]].merge(pd.DataFrame(rows), on=["Material ID", "Formula"])
    return out


def get_bartel():
    import hashlib
    for fn in BARTEL_FILES:
        if not os.path.exists(fn):
            urllib.request.urlretrieve(BARTEL_RAW + fn, fn)
        with open(fn, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != BARTEL_SHA256[fn]:
            raise RuntimeError(f"{fn} differs from the version used in the paper (SHA-256 {digest})")
    sys.path.insert(0, os.getcwd())
    from PredictPerovskites import PredictABX3  # noqa: E402
    return PredictABX3, pd.read_csv("TableS1.csv")


def bartel_descriptors(PredictABX3, formulas):
    rows, first = [], True
    for f in formulas:
        row = {"Formula": f}
        try:
            p = PredictABX3(f)
            if first:
                log("PredictABX3 attributes:", [a for a in dir(p) if not a.startswith("_")])
                first = False
            for attr, name in [("pred_A", "A"), ("pred_B", "B"), ("X", "X"), ("nA", "nA"), ("nB", "nB"),
                               ("rA", "rA"), ("rB", "rB"), ("rX", "rX"), ("t", "t"), ("tau", "tau"),
                               ("t_pred", "t_pred"), ("tau_pred", "tau_pred")]:
                try:
                    v = getattr(p, attr)
                    row[f"bartel_{name}"] = v if not isinstance(v, (list, dict)) else str(v)
                except Exception as exc:
                    row[f"bartel_{name}"] = np.nan
                    row.setdefault("bartel_error", f"{name}: {str(exc)[:60]}")
            try:
                row["bartel_mu"] = float(p.rB) / float(p.rX)
            except Exception:
                row["bartel_mu"] = np.nan
        except Exception as exc:
            row["bartel_error"] = str(exc)[:100]
        rows.append(row)
    return pd.DataFrame(rows)


def reduced(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return None


def main(check=False):
    raw = pd.read_csv("MP_ABO3_raw.csv", sep=";")
    with open("MP_ABO3_structures.json") as fh:
        structs = json.load(fh)
    log(f"entries: {len(raw)}, compositions: {raw.Formula.nunique()}")

    log("\n[1] structure-based sites, bond-valence states, topology")
    st = structure_part(raw, structs, check)
    st.to_csv("ABO3_topology_struct.csv" if not check else "check_topology_struct.csv", sep=";", index=False)
    log(f"  bond-valence failures: {int((~st.bv_ok.fillna(False).astype(bool)).sum())} / {len(st)}")
    log(f"  site rules: {st.site_rule.value_counts().to_dict()}")
    log(f"  topology: {st.topology_struct.value_counts().to_dict()}")
    if check:
        cols = ["Formula", "Material ID", "Space Group", "Energy Above Hull", "A_struct", "B_struct",
                "A_bv", "B_bv", "site_rule", "B_CN_O", "topology_struct", "t_struct"]
        log(st.sort_values(["Formula", "Energy Above Hull"])[[c for c in cols if c in st]].to_string(index=False))

    log("\n[2] Bartel reference implementation")
    try:
        PredictABX3, s1 = get_bartel()
        forms = sorted(st.Formula.unique())
        bd = bartel_descriptors(PredictABX3, forms)
        bd.to_csv("ABO3_descriptors_bartel.csv" if not check else "check_descriptors_bartel.csv", sep=";", index=False)
        n_err = bd["bartel_error"].notna().sum() if "bartel_error" in bd else 0
        log(f"  compositions: {len(bd)}, with errors: {n_err}")
        if check:
            log(bd.to_string(index=False))

        log("\n[3] overlap with Bartel's experimental set (oxides)")
        log(f"  TableS1 columns: {list(s1.columns)}")
        s1["red"] = s1["ABX3"].map(reduced)
        ox = s1[s1["ABX3"].astype(str).str.endswith("O3")].copy()
        st["red"] = st.Formula.map(reduced)
        common = sorted(set(ox.red) & set(st.red))
        log(f"  Bartel oxides: {len(ox)}, in MP ABO3 set: {len(common)}")
        icsd = st[st.Theoretical.astype(str).str.lower() == "false"].sort_values("Energy Above Hull")
        rows = []
        for red in common:
            q = icsd[icsd.red == red]
            rows.append({"reduced_formula": red,
                         "bartel_exp_label": int(ox[ox.red == red]["exp_label"].iloc[0]),
                         "n_icsd_entries": len(q),
                         "label_lowest_icsd": q.topology_struct.iloc[0] if len(q) else None,
                         "any_icsd_perovskite": bool((q.topology_struct == "perovskite").any()) if len(q) else None})
        pd.DataFrame(rows).to_csv("bartel_label_comparison.csv" if not check else "check_bartel_label_comparison.csv", sep=";", index=False)
        log(f"  written bartel_label_comparison.csv ({len(rows)} rows)")
    except Exception as exc:
        log(f"  Bartel part failed: {exc}")

    with open("04_run_report.txt" if not check else "04_check_report.txt", "w") as fh:
        fh.write("\n".join(REPORT))


if __name__ == "__main__":
    main(check="--check" in sys.argv)
