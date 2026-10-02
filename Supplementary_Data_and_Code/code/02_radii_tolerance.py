"""
02_radii_tolerance.py -- oxidation states, Shannon radii, t and mu.

For each composition:
  * charge-balanced oxidation states from pymatgen Composition.oxi_state_guesses()
    (integer, positive cation states, O = -2); if none exists with common states,
    all states are allowed and the composition is flagged; the number of valid
    assignments is recorded (oxi_ambiguous) and the most probable one is used;
  * the larger cation (compared at the same coordination number) is the A site;
  * Shannon (1976) ionic radii as tabulated in pymatgen: B at CN VI (high spin where
    both spin states exist); A at CN XII or the next lower tabulated CN
    (XII, XI, X, IX, VIII, VII, VI), CN used recorded per entry;
  * t and mu with r_O = 1.40 A; admissible if 0.80 <= t <= 1.10 and
    0.414 <= mu <= 0.732; t also at fixed A-site CN XII, IX, VIII, VI;
  * lone-pair B cations flagged (not removed).
Output: ABO3_radii_t_mu.csv
"""

import math

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Species

IN_CSV = "MP_ABO3_raw.csv"
OUT_CSV = "ABO3_radii_t_mu.csv"

R_O = 1.40                      # Shannon O2-, CN VI
T_WINDOW = (0.80, 1.10)
MU_WINDOW = (0.414, 0.732)
A_CN_ORDER = ["XII", "XI", "X", "IX", "VIII", "VII", "VI"]  # documented fallback
SENS_CNS = ["XII", "IX", "VIII", "VI"]                      # fixed-CN sensitivity
LONE_PAIR = {("As", 3), ("Sb", 3), ("Bi", 3), ("Se", 4), ("Te", 4), ("I", 5),
             ("Sn", 2), ("Pb", 2), ("Tl", 1), ("P", 3), ("S", 4)}


def shannon(el, ox, cn, spin_pref="High Spin"):
    """Shannon ionic radius or (None, None). Spin-independent ions use spin=''."""
    other = "Low Spin" if spin_pref == "High Spin" else "High Spin"
    sp = Species(el, ox)
    for spin in ("", spin_pref, other):
        try:
            return float(sp.get_shannon_radius(cn=cn, spin=spin, radius_type="ionic")), spin
        except Exception:
            continue
    return None, None


def a_site_radius(el, ox):
    for cn in A_CN_ORDER:
        r, _ = shannon(el, ox, cn)
        if r is not None:
            return r, cn
    return None, None


def tol(ra, rb):
    return (ra + R_O) / (math.sqrt(2) * (rb + R_O))


def compare_size(c1, o1, c2, o2):
    """Return (A, B) by comparing radii at the first CN available for BOTH ions."""
    for cn in ("VI", "VIII", "XII"):
        r1, _ = shannon(c1, o1, cn)
        r2, _ = shannon(c2, o2, cn)
        if r1 is not None and r2 is not None:
            return ((c1, o1), (c2, o2)) if r1 >= r2 else ((c2, o2), (c1, o1))
    return None


def process(formula):
    comp = Composition(formula)
    cations = [e.symbol for e in comp.elements if e.symbol != "O"]
    if len(cations) != 2:
        return {"exclusion_reason": "not two cations"}

    valid, uncommon = [], False
    for all_ox in (False, True):        # 2nd pass only if common states fail (e.g. Fe4+, Os5+)
        if valid:
            break
        uncommon = all_ox
        valid = _assignments(comp, cations, all_ox)

    if not valid:
        return {"exclusion_reason": "no charge-balanced assignment with tabulated Shannon radii"}
    return _row(valid, uncommon)


def _assignments(comp, cations, all_ox):
    valid = []
    for g in comp.oxi_state_guesses(all_oxi_states=all_ox):
        if g.get("O") != -2:
            continue
        ox = {c: g[c] for c in cations}
        if any(abs(v - round(v)) > 1e-6 or v <= 0 for v in ox.values()):
            continue                                   # mixed/negative valence
        ox = {c: int(round(v)) for c, v in ox.items()}
        pair = compare_size(cations[0], ox[cations[0]], cations[1], ox[cations[1]])
        if pair is None:
            continue
        (a, a_ox), (b, b_ox) = pair
        ra, a_cn = a_site_radius(a, a_ox)
        rb, b_spin = shannon(b, b_ox, "VI")
        if ra is None or rb is None:
            continue
        valid.append((a, a_ox, ra, a_cn, b, b_ox, rb, b_spin))
    return valid


def _row(valid, uncommon):
    a, a_ox, ra, a_cn, b, b_ox, rb, b_spin = valid[0]   # most probable guess
    t, mu = tol(ra, rb), rb / R_O
    out = {
        "Atom A": a, "A_ox": a_ox, "rA": ra, "A_CN_used": a_cn,
        "A_CN_fallback": a_cn != "XII",
        "Atom B": b, "B_ox": b_ox, "rB": rb, "B_spin": b_spin or "n/a",
        "t": round(t, 4), "mu": round(mu, 4),
        "n_oxi_assignments": len(valid),
        "oxi_ambiguous": len({(v[1], v[5]) for v in valid}) > 1,
        "B_lone_pair": (b, b_ox) in LONE_PAIR,
        "uncommon_oxidation_state": uncommon,
        "admissible": (T_WINDOW[0] <= t <= T_WINDOW[1]) and (MU_WINDOW[0] <= mu <= MU_WINDOW[1]),
        "exclusion_reason": "",
    }
    # Fixed-CN sensitivity columns (NaN where Shannon has no value at that CN)
    for cn in SENS_CNS:
        r, _ = shannon(a, a_ox, cn)
        out[f"t_A{cn}"] = round(tol(r, rb), 4) if r is not None else np.nan
    # Low-spin B variant where it differs
    r_ls, spin_ls = shannon(b, b_ox, "VI", spin_pref="Low Spin")
    out["t_B_lowspin"] = round(tol(ra, r_ls), 4) if spin_ls == "Low Spin" else np.nan
    return out


def in_window(t, mu):
    return t.between(*T_WINDOW) & mu.between(*MU_WINDOW)


if __name__ == "__main__":
    df = pd.read_csv(IN_CSV, sep=";")
    cache = {f: process(f) for f in df["Formula"].unique()}   # one call per composition
    res = pd.concat([df, pd.DataFrame([cache[f] for f in df["Formula"]], index=df.index)], axis=1)
    res.to_csv(OUT_CSV, index=False, sep=";")

    ok = res[res["exclusion_reason"] == ""]
    print(f"Entries: {len(res)} | with valid assignment: {len(ok)} "
          f"({ok['Formula'].nunique()} compositions)")
    print("Exclusion reasons:\n", res.loc[res.exclusion_reason != "", "exclusion_reason"].value_counts())
    print(f"A-site CN used:\n{ok.drop_duplicates('Formula')['A_CN_used'].value_counts()}")
    print(f"Ambiguous oxidation-state compositions: {ok.drop_duplicates('Formula')['oxi_ambiguous'].sum()}")

    # Radius-convention sensitivity table (for Section 3.2)
    print("\nAdmissible compositions by A-site radius convention:")
    base = set(ok.loc[ok["admissible"], "Formula"])
    print(f"  fallback order {A_CN_ORDER[0]}->...: {len(base)}")
    for cn in SENS_CNS:
        col = f"t_A{cn}"
        sub = ok[ok[col].notna()]
        adm = set(sub.loc[in_window(sub[col], sub["mu"]), "Formula"])
        print(f"  fixed CN {cn:>4}: {len(adm):4d} admissible "
              f"(lost {len(base - adm)}, gained {len(adm - base)}; "
              f"{ok['Formula'].nunique() - sub['Formula'].nunique()} comps lack CN {cn})")
    print(f"\nAdmissible with lone-pair B (flagged, not removed): "
          f"{ok.loc[ok.admissible & ok.B_lone_pair, 'Formula'].nunique()}")
    print(f"Saved -> {OUT_CSV}")
