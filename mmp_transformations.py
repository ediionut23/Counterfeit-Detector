"""
MMP Transformations for Counterfeit Generation — Academically Defensible Set
=============================================================================
 
Each transformation is annotated with:
  - DIFFICULTY: subtle | moderate | obvious
  - CATEGORY:   bioisostere | low-effort-modification | scaffold-hop
  - RATIONALE:  why this represents a realistic counterfeit or close analog
  - CITATION:   primary reference where this MMP is documented
 
PRIMARY REFERENCES
------------------
[Meanwell2011]   Meanwell, N. A. "Synopsis of Some Recent Tactical Application of
                 Bioisosteres in Drug Design." J. Med. Chem. 2011, 54(8), 2529-2591.
                 DOI: 10.1021/jm1013693.
                 Canonical review of bioisosteric replacements in modern drug design.
 
[Patani1996]     Patani, G. A.; LaVoie, E. J. "Bioisosterism: A Rational Approach in
                 Drug Design." Chem. Rev. 1996, 96(8), 3147-3176.
                 DOI: 10.1021/cr950066q.
                 Foundational review of classical and non-classical bioisosteres.
 
[Pennington2017] Pennington, L. D.; Moustakas, D. T. "The Necessary Nitrogen Atom:
                 A Versatile High-Impact Design Element for Multiparameter
                 Optimization." J. Med. Chem. 2017, 60(9), 3552-3579.
                 DOI: 10.1021/acs.jmedchem.6b01807.
                 Authoritative source for aryl C→N walks.
 
[Muller2007]     Muller, K.; Faeh, C.; Diederich, F. "Fluorine in Pharmaceuticals:
                 Looking Beyond Intuition." Science 2007, 317(5846), 1881-1886.
                 DOI: 10.1126/science.1131943.
                 Source for H→F aryl substitution as an SAR tactic.
 
[Kumari2020]     Kumari, S.; Carmona, A. V.; Tiwari, A. K.; Trippier, P. C.
                 "Amide Bond Bioisosteres: Strategies, Synthesis, and Successes."
                 J. Med. Chem. 2020, 63(21), 12290-12358. DOI: 10.1021/acs.jmedchem.0c00530.
 
SCOPE NOTE — WHAT THIS SET MODELS
----------------------------------
The "counterfeit" class in this dataset is deliberately broader than strict
Matched Molecular Pairs. It combines two complementary threat models:
 
  (1) STRUCTURED BIOISOSTERIC SUBSTITUTIONS — transformations that a
      sophisticated bad actor with medicinal-chemistry knowledge might
      apply to evade patent protection while preserving (or plausibly
      claiming) similar bioactivity. These are drawn directly from
      [Meanwell2011] and [Patani1996].
 
  (2) LOW-EFFORT STRUCTURAL MODIFICATIONS — single-atom or single-group
      substitutions (H→F, H→Cl, H→Me, halogen walks) that require no
      medicinal-chemistry expertise and model the much more common
      "cosmetic modification" pattern seen in counterfeit pharmaceutical
      reports. These substitutions produce molecules that are chemically
      real (all products are sanitizable, druglike, and novel) but whose
      biological equivalence to the parent is not claimed.
 
Every transformation listed here:
  * produces an RDKit-sanitizable molecule (verified empirically),
  * uses SMARTS with explicit atom mapping so the unmodified portion
    of the parent molecule is preserved,
  * has precedent in the cited literature OR is explicitly flagged as
    a low-effort modification (not a bioisostere claim).
 
Transformations we deliberately EXCLUDED from earlier versions:
  * ketone_to_sulfoxide — pseudo-bioisosteric; the geometric and
    electronic change is too large to treat as a substitution.
  * CN_to_CCF3 — [Meanwell2011] lists cyano→tetrazole and cyano→amidine
    as bioisosteric pairs, but NOT cyano→trifluoromethyl. Excluded.
"""
 
from typing import List, Tuple
 
MMP_TRANSFORMATIONS: List[Tuple[str, str, str, str, str, str]] = [
 
    # ================================================================
    #  CATEGORY 1: Well-documented bioisosteres
    #  (patent-evasion pattern for sophisticated counterfeiters)
    # ================================================================
 
    # --- SUBTLE bioisosteres ---
    # Aryl C→N walk: one of the most studied modifications in medicinal
    # chemistry. Phenyl → pyridyl isomers change H-bonding, basicity,
    # and solubility while keeping overall shape. [Pennington2017]
    ("phenyl_to_2pyridyl",       "subtle",   "bioisostere",
     "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[*:7]",
     "[n:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[*:7]",
     "Pennington2017"),
 
    ("phenyl_to_3pyridyl",       "subtle",   "bioisostere",
     "[c:1]1[cH:2][cH:3][c:4]([*:7])[cH:5][cH:6]1",
     "[c:1]1[cH:2][n:3][c:4]([*:7])[cH:5][cH:6]1",
     "Pennington2017"),
 
    # Alkyl chain extension on O. [Meanwell2011, Section 2]
    ("OMe_to_OEt",               "subtle",   "bioisostere",
     "[O:1]([CH3:2])[*:3]",
     "[O:1]([CH2:2][CH3])[*:3]",
     "Meanwell2011"),
 
    # Methyl→ethyl on arene. "Methyl/ethyl walk" — classic SAR probe.
    # [Patani1996, Section II.A]
    ("methyl_to_ethyl_on_aryl",  "subtle",   "bioisostere",
     "[c:1][CH3:2]",
     "[c:1][CH2:2][CH3]",
     "Patani1996"),
 
    # --- MODERATE bioisosteres ---
    # COOH → tetrazole: the textbook carboxylic-acid bioisostere.
    # Exemplified by losartan, valsartan. [Meanwell2011, Section 5;
    # Patani1996, Table 17]
    ("COOH_to_tetrazole",        "moderate", "bioisostere",
     "[C:1](=[O:2])[OH:3]",
     "[c:1]1[nH:2][n:3][n][n]1",
     "Meanwell2011"),
 
    # Methyl ester → N-methyl amide: stability against esterase hydrolysis.
    # Classic amide bond bioisostere. [Kumari2020]
    ("ester_to_amide",           "moderate", "bioisostere",
     "[C:1](=[O:2])[O:3][CH3:4]",
     "[C:1](=[O:2])[N:3][CH3:4]",
     "Kumari2020"),
 
    # Amide → sulfonamide bioisostere of the amide bond.
    # [Kumari2020, Section 4.2]
    ("amide_to_sulfonamide",     "moderate", "bioisostere",
     "[C:1](=[O:2])[NH:3][*:4]",
     "[S:1](=[O:2])(=O)[NH:3][*:4]",
     "Kumari2020"),
 
    # N-methylation of anilines: slows metabolism, modifies H-bond donor
    # character. [Meanwell2011, Section 2.4]
    ("ArNH2_to_ArNHMe",          "moderate", "bioisostere",
     "[c:1][NH2:2]",
     "[c:1][NH:2][CH3]",
     "Meanwell2011"),
 
    # Aryl-Cl → aryl-CF3. Lipophilic bioisostere widely used to block
    # metabolic oxidation at the halogen position. [Muller2007]
    ("ArCl_to_ArCF3",            "moderate", "bioisostere",
     "[c:1][Cl:2]",
     "[c:1][C:2](F)(F)F",
     "Muller2007"),
 
    # --- OBVIOUS bioisosteres (ring-level) ---
    # Piperidine → morpholine: oxygen replaces CH2 in saturated 6-ring.
    # Reduces basicity, increases polarity. [Meanwell2011, Section 3;
    # Patani1996, Section III.B]
    ("piperidine_to_morpholine", "obvious",  "bioisostere",
     "[N:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1",
     "[N:1]1[CH2:2][CH2:3][O][CH2:5][CH2:6]1",
     "Meanwell2011"),
 
    # Five-membered aromatic heterocycle swaps. [Patani1996, Section II.B]
    ("pyrrole_to_furan",         "obvious",  "bioisostere",
     "[nH:1]1[cH:2][cH:3][cH:4][cH:5]1",
     "[o:1]1[cH:2][cH:3][cH:4][cH:5]1",
     "Patani1996"),
 
    ("pyrrole_to_thiophene",     "obvious",  "bioisostere",
     "[nH:1]1[cH:2][cH:3][cH:4][cH:5]1",
     "[s:1]1[cH:2][cH:3][cH:4][cH:5]1",
     "Patani1996"),
 
    # ================================================================
    #  CATEGORY 2: Low-effort structural modifications
    #  (patent-evasion pattern for naive counterfeiters; NOT claimed
    #   to be biologically equivalent)
    # ================================================================
 
    # Halogen walks on arenes. These are the most common "cosmetic"
    # modifications reported in counterfeit pharmaceutical analyses.
    # Chemically valid and druglike, but biological activity can
    # differ substantially. [Muller2007]
    ("ArF_to_ArCl",              "subtle",   "low-effort",
     "[c:1][F:2]",
     "[c:1][Cl:2]",
     "Muller2007"),
 
    ("ArBr_to_ArCl",             "subtle",   "low-effort",
     "[c:1][Br:2]",
     "[c:1][Cl:2]",
     "Muller2007"),
 
    # Aromatic H→X substitutions. These model the "add/remove one atom"
    # modification commonly seen in counterfeit products. Each produces
    # a chemically real molecule. The counterfeit class rationale is
    # structural similarity, not claimed bioequivalence. [Muller2007]
    ("ArH_to_ArF",               "subtle",   "low-effort",
     "[cH:1]",
     "[c:1][F]",
     "Muller2007"),
 
    ("ArH_to_ArCl",              "subtle",   "low-effort",
     "[cH:1]",
     "[c:1][Cl]",
     "Muller2007"),
 
    ("ArH_to_ArMe",              "subtle",   "low-effort",
     "[cH:1]",
     "[c:1][CH3]",
     "Patani1996"),
 
    ("ArH_to_ArOH",              "subtle",   "low-effort",
     "[cH:1]",
     "[c:1][OH]",
     "Patani1996"),
 
    ("ArH_to_ArOMe",             "subtle",   "low-effort",
     "[cH:1]",
     "[c:1][O][CH3]",
     "Meanwell2011"),
 
    # ================================================================
    #  CATEGORY 3: Structural analogs (not strict bioisosteres)
    #  Mark these as OBVIOUS because they produce clearly
    #  distinguishable molecules.
    # ================================================================
 
    # Ring size changes. [Patani1996, Section III]
    ("cyclopentane_to_cyclohexane","obvious", "scaffold-hop",
     "[C:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1",
     "[C:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2]1",
     "Patani1996"),
 
    # Alkene hydrogenation. Produces a saturated analog — geometry
    # changes significantly but connectivity is preserved.
    ("alkene_to_alkane",         "obvious",  "scaffold-hop",
     "[CH:1]=[CH:2]",
     "[CH2:1][CH2:2]",
     "Patani1996"),
]
 
 
# Short human-readable descriptions useful for stats output.
TRANSFORMATION_DESCRIPTIONS = {
    "phenyl_to_2pyridyl":        "Phenyl → 2-pyridyl (C→N walk at ortho position)",
    "phenyl_to_3pyridyl":        "Phenyl → 3-pyridyl (C→N walk at meta position)",
    "OMe_to_OEt":                "Methoxy → ethoxy (chain extension)",
    "methyl_to_ethyl_on_aryl":   "Aryl methyl → aryl ethyl (chain extension)",
    "COOH_to_tetrazole":         "Carboxylic acid → 1H-tetrazole (classical COOH bioisostere)",
    "ester_to_amide":            "Methyl ester → N-methyl amide (stability bioisostere)",
    "amide_to_sulfonamide":      "Amide → sulfonamide (amide-bond bioisostere)",
    "ArNH2_to_ArNHMe":           "Aniline → N-methylaniline (metabolism-blocking)",
    "ArCl_to_ArCF3":             "Aryl chloride → aryl CF3 (lipophilic bioisostere)",
    "piperidine_to_morpholine":  "Piperidine → morpholine (basicity reduction)",
    "pyrrole_to_furan":          "Pyrrole → furan (5-ring heterocycle swap)",
    "pyrrole_to_thiophene":      "Pyrrole → thiophene (5-ring heterocycle swap)",
    "ArF_to_ArCl":               "Aryl F → aryl Cl (halogen walk)",
    "ArBr_to_ArCl":              "Aryl Br → aryl Cl (halogen walk)",
    "ArH_to_ArF":                "Aromatic C-H → C-F (fluorination)",
    "ArH_to_ArCl":               "Aromatic C-H → C-Cl (chlorination)",
    "ArH_to_ArMe":               "Aromatic C-H → C-CH3 (methylation)",
    "ArH_to_ArOH":                "Aromatic C-H → C-OH (hydroxylation)",
    "ArH_to_ArOMe":              "Aromatic C-H → C-OCH3 (methoxylation)",
    "cyclopentane_to_cyclohexane":"Cyclopentane → cyclohexane (ring expansion)",
    "alkene_to_alkane":          "Alkene → alkane (saturation)",
}
 
 
def get_transformation_summary() -> dict:
    """Return a summary dict useful for paper/thesis writing."""
    by_difficulty = {"subtle": 0, "moderate": 0, "obvious": 0}
    by_category = {"bioisostere": 0, "low-effort": 0, "scaffold-hop": 0}
    by_citation = {}
    for (name, diff, cat, _, _, cite) in MMP_TRANSFORMATIONS:
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
        by_citation[cite] = by_citation.get(cite, 0) + 1
    return {
        "total": len(MMP_TRANSFORMATIONS),
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "by_citation": by_citation,
    }
 
 
if __name__ == "__main__":
    import json
    s = get_transformation_summary()
    print("Transformation set summary:")
    print(json.dumps(s, indent=2))
    print("\nAll transformations:")
    for (name, diff, cat, _, _, cite) in MMP_TRANSFORMATIONS:
        desc = TRANSFORMATION_DESCRIPTIONS.get(name, "")
        print(f"  [{diff:8s}] [{cat:13s}] {name:30s} [{cite}]  {desc}")
 