"""
01_fetch_mp.py -- retrieval of ABO3 entries from the Materials Project.

Queries all ternary oxides through the MP API, keeps entries whose reduced
composition is A1B1O3 (two distinct cations, three O), and saves
  MP_ABO3_raw.csv            entry data (energy above hull, volume, space group,
                             MP `theoretical` flag = no matching ICSD structure);
                             missing values are kept as NaN
  MP_ABO3_structures.json    relaxed structures (pymatgen dicts)
  MP_retrieval_metadata.json database version, retrieval date, package versions, counts

Run:  export MP_API_KEY=...   then   python 01_fetch_mp.py
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

OUT_CSV = "MP_ABO3_raw.csv"
OUT_STRUCT = "MP_ABO3_structures.json"
OUT_META = "MP_retrieval_metadata.json"

FIELDS = [
    "material_id", "formula_pretty", "band_gap", "energy_above_hull",
    "volume", "density", "nsites", "symmetry", "theoretical", "structure",
]


def is_abo3(formula: str) -> bool:
    """True if the reduced composition is exactly A1 B1 O3 (two distinct cations)."""
    red = Composition(formula).reduced_composition
    if len(red.elements) != 3 or red["O"] != 3:
        return False
    return all(red[el] == 1 for el in red.elements if el.symbol != "O")


def _num(x):
    return float(x) if x is not None else np.nan


def fetch(api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("Set the MP_API_KEY environment variable first.")

    with MPRester(api_key) as mpr:
        try:
            db_version = getattr(mpr, "db_version", None) or mpr.get_database_version()
        except Exception as exc:  # record, do not crash
            db_version = f"unavailable ({exc})"
        print(f"Materials Project database version: {db_version}")
        docs = mpr.materials.summary.search(
            elements=["O"], num_elements=3, fields=FIELDS
        )
    print(f"Downloaded {len(docs)} ternary oxides; filtering to ABO3 ...")

    rows, structures = [], {}
    for d in docs:
        if not is_abo3(d.formula_pretty):
            continue
        mid = str(d.material_id)
        sym = d.symmetry
        cs = getattr(sym.crystal_system, "value", sym.crystal_system) if sym else None
        rows.append({
            "Material ID": mid,
            "Formula": d.formula_pretty,
            "Band Gap": _num(d.band_gap),
            "Energy Above Hull": _num(d.energy_above_hull),
            "Volume": _num(d.volume),
            "Density": _num(d.density),
            "Sites": int(d.nsites) if d.nsites is not None else np.nan,
            "Volume per atom": _num(d.volume) / d.nsites if d.volume and d.nsites else np.nan,
            "Crystal System": str(cs) if cs is not None else None,
            "Space Group": sym.symbol if sym else None,
            # MP meaning: True = no matching ICSD entry. NOT a synthesizability prediction.
            "Theoretical": bool(d.theoretical) if d.theoretical is not None else None,
        })
        if d.structure is not None:
            structures[mid] = d.structure.as_dict()

    df = pd.DataFrame(rows)
    n_missing_ehull = df["Energy Above Hull"].isna().sum()
    if n_missing_ehull:
        print(f"WARNING: {n_missing_ehull} entries have no E_hull (kept as NaN, not 0).")

    df.to_csv(OUT_CSV, index=False, sep=";")
    with open(OUT_STRUCT, "w") as fh:
        json.dump(structures, fh)

    from importlib.metadata import version as pkg_version
    meta = {
        "retrieval_date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "mp_database_version": db_version,
        "mp_api_version": pkg_version("mp-api"),
        "pymatgen_version": pkg_version("pymatgen"),
        "query": {"elements": ["O"], "num_elements": 3},
        "post_filter": "reduced composition A1B1O3",
        "energy_note": ("energy_above_hull from the summary endpoint (default thermodynamic "
                        "scheme of this database version)"),
        "n_ternary_oxides": len(docs),
        "n_abo3_entries": len(df),
        "n_abo3_compositions": int(df["Formula"].nunique()),
        "n_structures_saved": len(structures),
    }
    with open(OUT_META, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved {len(df)} ABO3 entries ({df['Formula'].nunique()} compositions) -> {OUT_CSV}")
    print(f"Saved {len(structures)} structures -> {OUT_STRUCT}")
    print(f"Provenance -> {OUT_META}")
    return df


if __name__ == "__main__":
    fetch(os.environ.get("MP_API_KEY"))
