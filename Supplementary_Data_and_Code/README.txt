Supplementary Material
"Benchmarking Tolerance-Factor Screening in ABO3 Oxides"
K. Drakakis, V. Kochliaridis, C. Apokatanidis, I. Pavli

Materials Project database version 2026.04.13 (retrieved 2 October 2026); pymatgen 2026.9.23.
The mp-api version was not recorded by the retrieval run (the field reads "unknown" in
MP_retrieval_metadata.json); script 01 now records it, together with the pymatgen version.

code/
  01_fetch_mp.py           Retrieval of all A1B1O3 entries and structures (needs MP_API_KEY).
  02_radii_tolerance.py    Shannon-radius helper functions; script 04 uses them only for an
                           auxiliary column (t_struct) that is not used in the paper.
  03_topology.py           CrystalNN-based BO6 connectivity and A-site cavity classifier
                           (its classify() function is called by script 04).
  04_structure_labels_and_bartel.py
                           Structure-based A/B site assignment (coordination and bond valence),
                           topology label of every entry; t, mu and tau from the reference
                           implementation of Bartel et al. (downloaded, with their experimental
                           labels TableS1.csv, from github.com/CJBartel/perovskite-stability);
                           overlap with their labels. Needs pymatgen, scikit-learn, internet.
                           The repository has no tagged release, so the four files used for the
                           paper are pinned by SHA-256 in the script (a mismatch stops the run):
                             PredictPerovskites.py    2a33a186b1dd75281f8310f2bbf465a73214776a791dd411d9a530bad3167446
                             TableS1.csv              bea471ddb1ed63dcb94cc9cbc58c73f3bc1a12918e96cf24818b3aea2cbe3656
                             Shannon_radii_dict.json  322870e5a1fb45e168b4ff6c8e262f88122fe5c662b1f9c5cc9676ecb74b425f
                             electronegativities.csv  94e26be5fd3fcd156b0c71dd8588c61947289d7e25cdeece3c277a3df00255b8
  05_analysis.py           Every number in the paper (Tables 1-6). pandas only.
  06_figures.py            Figures 1-3 (vector PDF, TIFF 1000 dpi, PNG 600 dpi). matplotlib.

Run order: 01 -> 04 (--check first) -> 05 -> 06.
To reproduce the paper from data/ without an API key: put code/* and data/* in one folder,
obtain TableS1.csv from the repository above (script 04 downloads it), then run 05 and 06.

data/
  MP_retrieval_metadata.json        retrieval provenance
  MP_ABO3_raw.csv                   2,563 entries (energy above hull in eV/atom, volume per atom,
                                    space group, MP "theoretical" flag: False = ICSD-matched)
  MP_ABO3_structures.json           relaxed structures (pymatgen dictionaries)
  ABO3_topology_struct.csv          output of 04, one row per entry
  ABO3_descriptors_bartel.csv       output of 04, one row per composition (Bartel's code returns
                                    no descriptors for 366 compositions without a valid assignment;
                                    file regenerated with the final version of the code: in the
                                    original run the A/B attributes were requested under the wrong
                                    names, which did not affect any numerical value)
  bartel_label_comparison.csv       output of 04
  ABO3_labels_and_descriptors.csv   output of 05, one row per labelled composition
  05_analysis_output.txt            console report of 05 (all numbers in the paper)

CSV separator: semicolon (;).
