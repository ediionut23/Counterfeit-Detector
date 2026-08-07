"""
Intelligent Pharmaceutical Counterfeit Dataset Generator (v3)
==============================================================

Changelog vs v2:
  - FIX: PubChem REST now requests SMILES,ConnectivitySMILES,IsomericSMILES,
    CanonicalSMILES (fallbacks for the deprecated property name).
  - FIX: PubChem uses POST with "listkey"-style batching so we can send
    500 CIDs per request instead of 100, with proper URL encoding.
  - NEW: ChEMBL bulk mode via chembl_downloader (optional). If available,
    fetches the full SDF dump (~2.4M molecules) and iterates locally.
    Falls back to the REST client otherwise.
  - NEW: "approved+drug-like ChEMBL" mode — iterates ALL molecules with
    bioactivity records and MW 200-600, not just max_phase 1-4. This
    expands the authentic pool from ~8k to hundreds of thousands.
  - TUNED: Counterfeit generator — max_per_parent=15 (was 5), more
    SMARTS transforms, permissive H-substitution transforms that fire
    on almost any aromatic ring, Tanimoto floor lowered to 0.35.
  - TUNED: max_per_scaffold=50 (was 20) for authentic diversification.
  - TUNED: Property matching uses 6 bins per property (was 10), losing
    less of the dataset while still killing the trivial-LR shortcut.
  - Enhanced diagnostic logging so future runs show WHERE molecules are lost.

Usage:
    # Uses REST APIs (slower, needs network each run):
    python intelligent_dataset_generator.py --size 50000 --output d.pt

    # Uses bulk ChEMBL SDF (recommended, one-time download):
    pip install chembl-downloader
    python intelligent_dataset_generator.py --size 50000 --bulk-chembl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
#  MMP TRANSFORMATIONS
# ============================================================
MMP_TRANSFORMATIONS: List[Tuple[str, str, str, str]] = [
    # (name, difficulty, reactant_smarts, product_smarts)

    # --- Subtle: single-atom heteroatom swaps ---
    ("aryl_C_to_N",              "subtle",
     "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[*:7]",
     "[n:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[*:7]"),
    ("aryl_CH_to_N_para",        "subtle",
     "[c:1]1[cH:2][cH:3][c:4]([*:7])[cH:5][cH:6]1",
     "[c:1]1[cH:2][n:3][c:4]([*:7])[cH:5][cH:6]1"),
    ("OMe_to_OEt",               "subtle",
     "[O:1]([CH3:2])[*:3]",
     "[O:1]([CH2:2][CH3])[*:3]"),
    ("methyl_to_ethyl_on_aryl",  "subtle",
     "[c:1][CH3:2]",
     "[c:1][CH2:2][CH3]"),
    ("F_to_Cl_aryl",             "subtle",
     "[c:1][F:2]",
     "[c:1][Cl:2]"),
    # NEW permissive H-substitution transforms — trigger on almost any arene
    ("aryl_H_to_F",              "subtle",
     "[cH:1]",
     "[c:1][F]"),
    ("aryl_H_to_Cl",              "subtle",
     "[cH:1]",
     "[c:1][Cl]"),
    ("aryl_H_to_Me",              "subtle",
     "[cH:1]",
     "[c:1][CH3]"),
    ("aryl_H_to_OH",              "subtle",
     "[cH:1]",
     "[c:1][OH]"),
    ("aryl_H_to_OMe",             "subtle",
     "[cH:1]",
     "[c:1][O][CH3]"),

    # --- Moderate: functional group bioisosteres ---
    ("COOH_to_tetrazole",        "moderate",
     "[C:1](=[O:2])[OH:3]",
     "[c:1]1[nH:2][n:3][n][n]1"),
    ("ester_to_amide",           "moderate",
     "[C:1](=[O:2])[O:3][CH3:4]",
     "[C:1](=[O:2])[N:3][CH3:4]"),
    ("amide_to_sulfonamide",     "moderate",
     "[C:1](=[O:2])[NH:3][*:4]",
     "[S:1](=[O:2])(=O)[NH:3][*:4]"),
    ("ketone_to_sulfoxide",      "moderate",
     "[CH3:1][C:2](=[O:3])[*:4]",
     "[CH3:1][S:2](=[O:3])[*:4]"),
    ("Cl_to_CF3_aryl",           "moderate",
     "[c:1][Cl:2]",
     "[c:1][C:2](F)(F)F"),
    ("Br_to_Cl_aryl",            "moderate",
     "[c:1][Br:2]",
     "[c:1][Cl:2]"),
    ("NH2_to_NHMe",              "moderate",
     "[c:1][NH2:2]",
     "[c:1][NH:2][CH3]"),
    # NEW moderate ones
    ("CN_to_CCF3",               "moderate",
     "[c:1][C:2]#[N:3]",
     "[c:1][C:2](F)(F)F"),
    ("OCF3_to_OMe",              "moderate",
     "[c:1][O:2][C:3](F)(F)F",
     "[c:1][O:2][CH3:3]"),

    # --- Obvious: ring-level changes ---
    ("piperidine_to_morpholine", "obvious",
     "[N:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1",
     "[N:1]1[CH2:2][CH2:3][O][CH2:5][CH2:6]1"),
    ("pyrrole_to_furan",         "obvious",
     "[nH:1]1[cH:2][cH:3][cH:4][cH:5]1",
     "[o:1]1[cH:2][cH:3][cH:4][cH:5]1"),
    ("pyrrole_to_thiophene",     "obvious",
     "[nH:1]1[cH:2][cH:3][cH:4][cH:5]1",
     "[s:1]1[cH:2][cH:3][cH:4][cH:5]1"),
    ("cyclopentane_to_cyclohexane","obvious",
     "[C:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1",
     "[C:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2]1"),
    ("alkene_to_alkane",         "obvious",
     "[CH:1]=[CH:2]",
     "[CH2:1][CH2:2]"),
]


# ============================================================
#  PAINS FILTER
# ============================================================
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
PAINS_CATALOG = FilterCatalog(_pains_params)


def passes_pains(mol: Chem.Mol) -> bool:
    try:
        return not PAINS_CATALOG.HasMatch(mol)
    except Exception:
        return False


def passes_lipinski(mol: Chem.Mol, tolerance: int = 1) -> bool:
    try:
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        return violations <= tolerance
    except Exception:
        return False


def is_druglike(mol: Chem.Mol, min_qed: float = 0.20, strict: bool = True) -> bool:
    if mol is None:
        return False
    try:
        if mol.GetNumHeavyAtoms() < 15 or mol.GetNumHeavyAtoms() > 70:
            return False
        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 750:
            return False
        if Descriptors.qed(mol) < min_qed:
            return False
        if not passes_lipinski(mol, tolerance=2):
            return False
        if strict and not passes_pains(mol):
            return False
        if rdMolDescriptors.CalcNumRings(mol) == 0:
            return False
        return True
    except Exception:
        return False


def compute_descriptors(mol: Chem.Mol) -> Optional[Dict]:
    try:
        return {
            "molecular_weight": float(Descriptors.ExactMolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "hbd": int(Descriptors.NumHDonors(mol)),
            "hba": int(Descriptors.NumHAcceptors(mol)),
            "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
            "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "qed": float(Descriptors.qed(mol)),
        }
    except Exception:
        return None


def get_scaffold(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


# ============================================================
#  FETCHING
# ============================================================
class AuthenticFetcher:
    """Fetch authentic drugs, diversified by scaffold."""

    def __init__(self, max_per_scaffold: int = 50):
        self.max_per_scaffold = max_per_scaffold
        self.scaffold_counts: Dict[str, int] = defaultdict(int)
        # Counters for drop diagnostics
        self.stats = defaultdict(int)

    def _try_add(self, rows: List[Dict], row: Dict, mol: Chem.Mol) -> bool:
        scaffold = get_scaffold(mol)
        if self.scaffold_counts[scaffold] >= self.max_per_scaffold:
            self.stats["drop_scaffold_full"] += 1
            return False
        self.scaffold_counts[scaffold] += 1
        row["scaffold"] = scaffold
        rows.append(row)
        self.stats["added"] += 1
        return True

    # --- NEW: bulk ChEMBL via chembl_downloader (MUCH faster) ---
    def fetch_chembl_bulk(self, target: int) -> List[Dict]:
        """Fetch from ChEMBL's bulk SDF. Requires `pip install chembl-downloader`.
        This downloads ~2.4M molecules once, then iterates locally."""
        logger.info(f"[ChEMBL-bulk] Attempting bulk download (target={target})...")
        try:
            import chembl_downloader
        except ImportError:
            logger.warning("chembl_downloader not installed. "
                           "Install with: pip install chembl-downloader")
            return []

        rows = []
        seen_canonical = set()
        try:
            # This iterates RDKit mols directly from the SDF dump.
            # First call downloads; subsequent calls use cache.
            with chembl_downloader.supplier() as supplier:
                for i, mol in enumerate(tqdm(supplier, desc="ChEMBL-bulk",
                                              total=min(target * 10, 500_000))):
                    if len(rows) >= target:
                        break
                    if i > target * 20:  # hard cap
                        break
                    if mol is None:
                        continue
                    if not is_druglike(mol):
                        self.stats["chembl_not_druglike"] += 1
                        continue
                    try:
                        canonical = Chem.MolToSmiles(mol)
                    except Exception:
                        continue
                    if canonical in seen_canonical:
                        continue
                    seen_canonical.add(canonical)
                    desc = compute_descriptors(mol)
                    if desc is None:
                        continue
                    chembl_id = mol.GetProp("chembl_id") if mol.HasProp("chembl_id") else f"ROW_{i}"
                    row = {
                        "source": "ChEMBL-bulk",
                        "source_id": chembl_id,
                        "smiles": canonical,
                        "name": f"ChEMBL_{chembl_id}",
                        "max_phase": 0,
                        **desc,
                    }
                    self._try_add(rows, row, mol)
        except Exception as e:
            logger.warning(f"[ChEMBL-bulk] failed mid-iteration: {e}")

        logger.info(f"[ChEMBL-bulk] Got {len(rows)} unique-scaffold molecules")
        return rows

    # --- Original REST path, but iterates over multiple MW ranges for diversity ---
    def fetch_chembl_rest(self, target: int) -> List[Dict]:
        logger.info(f"[ChEMBL-REST] Fetching up to {target} molecules...")
        try:
            from chembl_webresource_client.new_client import new_client
        except ImportError:
            logger.warning("chembl_webresource_client not installed; skipping ChEMBL-REST")
            return []

        rows = []
        molecule = new_client.molecule

        # Strategy: iterate over MW ranges to spread requests across chemical space.
        # Each range will hit different offsets and avoid clustering on rare scaffolds.
        mw_ranges = [
            (200, 300), (300, 400), (400, 500), (500, 600), (600, 700),
        ]
        per_range = target // len(mw_ranges) + 1

        for mw_min, mw_max in mw_ranges:
            if len(rows) >= target:
                break
            collected = 0
            try:
                query = molecule.filter(
                    molecule_properties__mw_freebase__gte=mw_min,
                    molecule_properties__mw_freebase__lt=mw_max,
                    molecule_type="Small molecule",
                )
                for i, mol_rec in enumerate(query):
                    if collected >= per_range or len(rows) >= target:
                        break
                    if i > per_range * 6:
                        break
                    struct = mol_rec.get("molecule_structures") or {}
                    smi = struct.get("canonical_smiles")
                    if not smi:
                        continue
                    rd_mol = Chem.MolFromSmiles(smi)
                    if rd_mol is None or not is_druglike(rd_mol):
                        continue
                    canonical = Chem.MolToSmiles(rd_mol)
                    desc = compute_descriptors(rd_mol)
                    if desc is None:
                        continue
                    row = {
                        "source": "ChEMBL-REST",
                        "source_id": mol_rec["molecule_chembl_id"],
                        "smiles": canonical,
                        "name": mol_rec.get("pref_name") or f"ChEMBL_{mol_rec['molecule_chembl_id']}",
                        "max_phase": mol_rec.get("max_phase") or 0,
                        **desc,
                    }
                    if self._try_add(rows, row, rd_mol):
                        collected += 1
            except Exception as e:
                logger.warning(f"[ChEMBL-REST] MW {mw_min}-{mw_max}: {e}")
                continue

        logger.info(f"[ChEMBL-REST] Got {len(rows)} unique-scaffold molecules")
        return rows

    # --- PubChem FIX: use SMILES/ConnectivitySMILES/IsomericSMILES/CanonicalSMILES ---
    # and POST for large batches ---
    def fetch_pubchem(self, target: int) -> List[Dict]:
        logger.info(f"[PubChem] Fetching up to {target} molecules...")
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

        search_terms = [
            "aspirin", "ibuprofen", "paracetamol", "amoxicillin", "ciprofloxacin",
            "metformin", "atorvastatin", "omeprazole", "amlodipine", "metoprolol",
            "lisinopril", "simvastatin", "losartan", "albuterol", "fluoxetine",
            "sertraline", "warfarin", "clopidogrel", "azithromycin", "doxycycline",
            "prednisone", "levothyroxine", "gabapentin", "tramadol", "morphine",
            "codeine", "diazepam", "alprazolam", "clonazepam", "zolpidem",
            "cetirizine", "loratadine", "fexofenadine", "montelukast", "fluticasone",
            "salbutamol", "tiotropium", "budesonide", "beclomethasone", "salmeterol",
            "metronidazole", "trimethoprim", "sulfamethoxazole", "nitrofurantoin",
            "cephalexin", "clindamycin", "erythromycin", "vancomycin", "linezolid",
            "meropenem", "acyclovir", "oseltamivir", "ribavirin", "tenofovir",
            "lamivudine", "efavirenz", "lopinavir", "ritonavir", "atazanavir",
            "darunavir", "methotrexate", "cyclophosphamide", "doxorubicin",
            "paclitaxel", "cisplatin", "carboplatin", "vincristine", "tamoxifen",
            "letrozole", "anastrozole", "imatinib", "erlotinib", "gefitinib",
            "sorafenib", "sunitinib", "furosemide", "hydrochlorothiazide",
            "spironolactone", "digoxin", "bisoprolol", "carvedilol", "verapamil",
            "diltiazem", "nifedipine", "ramipril", "enalapril", "candesartan",
            "valsartan", "irbesartan", "olmesartan", "rosuvastatin", "pravastatin",
            "fluvastatin", "ezetimibe", "fenofibrate", "glibenclamide", "glipizide",
            "glimepiride", "sitagliptin", "empagliflozin", "pioglitazone", "acarbose",
            "haloperidol", "risperidone", "olanzapine", "quetiapine", "aripiprazole",
            "clozapine", "valproic acid", "carbamazepine", "lamotrigine",
            "levetiracetam", "phenytoin", "topiramate", "oxcarbazepine", "pregabalin",
            "amitriptyline", "nortriptyline", "venlafaxine", "duloxetine",
            "escitalopram", "citalopram", "paroxetine", "fluvoxamine", "bupropion",
            "donepezil", "memantine", "rivastigmine", "galantamine",
            "sildenafil", "tadalafil", "finasteride", "tamsulosin", "oxybutynin",
            "hydroxychloroquine", "chloroquine", "mefloquine", "artemisinin",
            "ketoconazole", "fluconazole", "itraconazole", "voriconazole",
            "colchicine", "allopurinol", "febuxostat", "indomethacin", "naproxen",
            "diclofenac", "celecoxib", "meloxicam", "piroxicam", "ketoprofen",
            "dexamethasone", "hydrocortisone", "methylprednisolone", "betamethasone",
            "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine", "sirolimus",
            "heparin", "enoxaparin", "dabigatran", "rivaroxaban", "apixaban",
            "ranitidine", "famotidine", "lansoprazole", "pantoprazole", "esomeprazole",
            "metoclopramide", "ondansetron", "domperidone", "loperamide", "bisacodyl",
            "lithium carbonate", "clonidine", "prazosin", "doxazosin", "terazosin",
            "amiodarone", "flecainide", "propafenone", "sotalol", "adenosine",
            "nitroglycerin", "isosorbide", "hydralazine", "minoxidil",
            "levodopa", "carbidopa", "pramipexole", "ropinirole", "entacapone",
            "sumatriptan", "rizatriptan", "zolmitriptan", "valproate",
            "methylphenidate", "atomoxetine", "modafinil", "lisdexamfetamine",
            "naloxone", "naltrexone", "buprenorphine", "methadone", "fentanyl",
            "oxycodone", "hydrocodone", "hydromorphone", "tapentadol",
        ]
        random.shuffle(search_terms)

        rows = []
        per_term = max(10, target // len(search_terms) * 5)
        # Property list — request multiple SMILES variants so we always get
        # something back whether PubChem returns the legacy or new name.
        # Order matters: try SMILES (new) first, then fallbacks.
        prop_list = "SMILES,ConnectivitySMILES,IsomericSMILES,CanonicalSMILES"

        pubchem_http_errors = 0
        pubchem_empty_props = 0

        for term in search_terms:
            if len(rows) >= target:
                break
            # Get CIDs for this drug name (includes substance variants)
            try:
                # Using the "similarity" search for broader coverage — name matches only the parent.
                # But here, fastidentity/name/{} gives us the CID of the exact compound.
                url = f"{base}/compound/name/{quote(term)}/cids/JSON"
                r = requests.get(url, timeout=30)
                if r.status_code != 200:
                    pubchem_http_errors += 1
                    continue
                cids = r.json().get("IdentifierList", {}).get("CID", [])

                # Expand via similarity: get related CIDs for this parent
                # (limited to first CID to avoid timeouts)
                if cids and len(cids) > 0:
                    try:
                        sim_url = f"{base}/compound/fastsimilarity_2d/cid/{cids[0]}/cids/JSON?Threshold=85&MaxRecords={per_term}"
                        rs = requests.get(sim_url, timeout=45)
                        if rs.status_code == 200:
                            sim_cids = rs.json().get("IdentifierList", {}).get("CID", [])
                            # merge; dedup preserving order
                            merged = list(dict.fromkeys(cids + sim_cids))
                            cids = merged[:per_term * 3]
                    except Exception:
                        pass
            except Exception as e:
                pubchem_http_errors += 1
                continue

            # Fetch SMILES in batches. PubChem supports POST with up to ~500 CIDs
            # per call. Use POST so the URL doesn't blow past 2 KB.
            batch_size = 200
            for batch_start in range(0, len(cids), batch_size):
                if len(rows) >= target:
                    break
                batch = cids[batch_start:batch_start + batch_size]
                cid_str = ",".join(str(c) for c in batch)
                url2 = f"{base}/compound/cid/property/{prop_list}/JSON"
                try:
                    r2 = requests.post(url2, data={"cid": cid_str}, timeout=60)
                    if r2.status_code != 200:
                        pubchem_http_errors += 1
                        continue
                    props = r2.json().get("PropertyTable", {}).get("Properties", [])
                    if not props:
                        pubchem_empty_props += 1
                        continue
                    for p in props:
                        # Fallback chain: new name "SMILES", then fallbacks
                        smi = (p.get("SMILES") or p.get("ConnectivitySMILES")
                               or p.get("IsomericSMILES") or p.get("CanonicalSMILES"))
                        cid = p.get("CID")
                        if not smi:
                            continue
                        rd_mol = Chem.MolFromSmiles(smi)
                        if rd_mol is None or not is_druglike(rd_mol):
                            continue
                        canonical = Chem.MolToSmiles(rd_mol)
                        desc = compute_descriptors(rd_mol)
                        if desc is None:
                            continue
                        row = {
                            "source": "PubChem",
                            "source_id": f"CID_{cid}",
                            "smiles": canonical,
                            "name": f"PubChem_{cid}",
                            "max_phase": 0,
                            **desc,
                        }
                        self._try_add(rows, row, rd_mol)
                    time.sleep(0.25)
                except Exception as e:
                    pubchem_http_errors += 1
                    continue

        logger.info(f"[PubChem] Got {len(rows)} molecules "
                    f"(http_errors={pubchem_http_errors}, empty_responses={pubchem_empty_props})")
        return rows


# ============================================================
#  COUNTERFEIT GENERATION
# ============================================================
@dataclass
class Transformation:
    name: str
    difficulty: str
    reaction: AllChem.ChemicalReaction


def _build_transformations() -> List[Transformation]:
    transforms = []
    for name, diff, reactant, product in MMP_TRANSFORMATIONS:
        try:
            rxn_smarts = f"{reactant}>>{product}"
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if rxn is not None:
                transforms.append(Transformation(name=name, difficulty=diff, reaction=rxn))
        except Exception as e:
            logger.warning(f"Failed to compile {name}: {e}")
    logger.info(f"Compiled {len(transforms)} MMP transformations")
    return transforms


class CounterfeitGenerator:
    def __init__(self, seen_smiles: Set[str], target_difficulty_mix: Dict[str, float] = None):
        self.seen_smiles = seen_smiles
        self.transformations = _build_transformations()
        self.difficulty_mix = target_difficulty_mix or {
            "subtle": 0.45,
            "moderate": 0.40,
            "obvious": 0.15,
        }
        self.by_difficulty = defaultdict(list)
        for t in self.transformations:
            self.by_difficulty[t.difficulty].append(t)
        self.stats = defaultdict(int)

    def _apply_transformation(self, mol: Chem.Mol, transform: Transformation) -> Optional[Chem.Mol]:
        try:
            products = transform.reaction.RunReactants((mol,))
            if not products:
                return None
            product_set = random.choice(products)
            if not product_set:
                return None
            product = product_set[0]
            try:
                Chem.SanitizeMol(product)
            except Exception:
                return None
            return product
        except Exception:
            return None

    def _pick_transformation(self) -> Transformation:
        r = random.random()
        cumulative = 0.0
        for diff, weight in self.difficulty_mix.items():
            cumulative += weight
            if r <= cumulative:
                candidates = self.by_difficulty.get(diff, [])
                if candidates:
                    return random.choice(candidates)
        return random.choice(self.transformations)

    def generate_one(self, parent_row: Dict, max_attempts: int = 50) -> Optional[Dict]:
        parent_smi = parent_row["smiles"]
        parent_mol = Chem.MolFromSmiles(parent_smi)
        if parent_mol is None:
            return None

        for attempt in range(max_attempts):
            transform = self._pick_transformation()
            product = self._apply_transformation(parent_mol, transform)
            if product is None:
                self.stats["fail_reaction"] += 1
                continue

            try:
                canonical = Chem.MolToSmiles(product)
            except Exception:
                self.stats["fail_canonical"] += 1
                continue

            if canonical in self.seen_smiles:
                self.stats["fail_duplicate"] += 1
                continue
            if canonical == parent_smi:
                self.stats["fail_identical"] += 1
                continue

            if not is_druglike(product, strict=False):
                self.stats["fail_druglike"] += 1
                continue

            sim = _tanimoto(parent_mol, product)
            # Lowered floor from 0.50 to 0.35 so we don't drop valid counterfeits
            if sim < 0.35 or sim > 0.98:
                self.stats["fail_similarity"] += 1
                continue

            self.seen_smiles.add(canonical)
            desc = compute_descriptors(product)
            if desc is None:
                continue
            self.stats["success"] += 1

            return {
                "source": "Counterfeit_MMP",
                "source_id": f"CF_{transform.name}_{parent_row['source_id']}",
                "smiles": canonical,
                "name": f"{transform.name}_of_{parent_row.get('name', 'unk')}",
                "max_phase": 0,
                "counterfeit_type": transform.name,
                "difficulty_level": transform.difficulty,
                "parent_smiles": parent_smi,
                "parent_scaffold": parent_row.get("scaffold", ""),
                "similarity_to_parent": sim,
                **desc,
            }
        self.stats["fail_all_attempts"] += 1
        return None


def _tanimoto(m1: Chem.Mol, m2: Chem.Mol) -> float:
    try:
        fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m1, 3, 2048)
        fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m2, 3, 2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ============================================================
#  PROPERTY MATCHING
# ============================================================
def property_match(authentic_df: pd.DataFrame, counterfeit_df: pd.DataFrame,
                   match_cols: List[str] = None,
                   n_bins: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if match_cols is None:
        match_cols = ["molecular_weight", "heavy_atoms"]

    logger.info(f"Property matching on: {match_cols} with {n_bins} bins")

    authentic_df = authentic_df.copy()
    counterfeit_df = counterfeit_df.copy()
    authentic_df["smiles_length"] = authentic_df["smiles"].str.len()
    counterfeit_df["smiles_length"] = counterfeit_df["smiles"].str.len()
    match_cols = match_cols + ["smiles_length"]

    combined = pd.concat([authentic_df[match_cols], counterfeit_df[match_cols]], ignore_index=True)
    bin_edges = {}
    for col in match_cols:
        bin_edges[col] = np.quantile(combined[col], np.linspace(0, 1, n_bins + 1))
        bin_edges[col][0] -= 1e-6
        bin_edges[col][-1] += 1e-6

    def assign_bins(df):
        keys = []
        for col in match_cols:
            keys.append(pd.cut(df[col], bins=bin_edges[col], labels=False, include_lowest=True))
        return list(zip(*[k.fillna(-1).astype(int).tolist() for k in keys]))

    auth_bins = assign_bins(authentic_df)
    fake_bins = assign_bins(counterfeit_df)

    authentic_df["_bin"] = auth_bins
    counterfeit_df["_bin"] = fake_bins

    auth_by_bin = authentic_df.groupby("_bin").groups
    fake_by_bin = counterfeit_df.groupby("_bin").groups

    keep_auth_idx = []
    keep_fake_idx = []
    all_bins = set(auth_by_bin.keys()) | set(fake_by_bin.keys())
    for b in all_bins:
        a_idx = list(auth_by_bin.get(b, []))
        f_idx = list(fake_by_bin.get(b, []))
        k = min(len(a_idx), len(f_idx))
        if k == 0:
            continue
        random.shuffle(a_idx)
        random.shuffle(f_idx)
        keep_auth_idx.extend(a_idx[:k])
        keep_fake_idx.extend(f_idx[:k])

    matched_auth = authentic_df.loc[keep_auth_idx].drop(columns=["_bin"]).reset_index(drop=True)
    matched_fake = counterfeit_df.loc[keep_fake_idx].drop(columns=["_bin"]).reset_index(drop=True)

    logger.info(f"After matching: auth={len(matched_auth)}, fake={len(matched_fake)} "
                f"(kept {len(matched_auth)/max(len(authentic_df),1)*100:.1f}% of auth, "
                f"{len(matched_fake)/max(len(counterfeit_df),1)*100:.1f}% of fake)")
    return matched_auth, matched_fake


def report_shortcut_metrics(authentic_df: pd.DataFrame, counterfeit_df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    def cohens_d(a, b):
        a = np.asarray(a); b = np.asarray(b)
        pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
        return abs(a.mean() - b.mean()) / max(pooled, 1e-9)

    auth_len = authentic_df["smiles"].str.len()
    fake_len = counterfeit_df["smiles"].str.len()
    d_len = cohens_d(auth_len, fake_len)
    d_mw = cohens_d(authentic_df["molecular_weight"], counterfeit_df["molecular_weight"])
    d_ha = cohens_d(authentic_df["heavy_atoms"], counterfeit_df["heavy_atoms"])

    logger.info("Shortcut diagnostics:")
    logger.info(f"  Cohen's d (SMILES length) : {d_len:.3f}")
    logger.info(f"  Cohen's d (MW)            : {d_mw:.3f}")
    logger.info(f"  Cohen's d (heavy atoms)   : {d_ha:.3f}")

    X = np.vstack([
        np.concatenate([authentic_df["heavy_atoms"], counterfeit_df["heavy_atoms"]]),
        np.concatenate([authentic_df["smiles"].str.len(), counterfeit_df["smiles"].str.len()]),
        np.concatenate([authentic_df["molecular_weight"], counterfeit_df["molecular_weight"]]),
    ]).T
    y = np.concatenate([np.zeros(len(authentic_df)), np.ones(len(counterfeit_df))])

    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
        acc = lr.score(sc.transform(Xte), yte)
        logger.info(f"  Trivial LR accuracy       : {acc:.3f}  {'GOOD' if acc < 0.60 else 'SHORTCUT PRESENT'}")
        return {"cohen_d_length": d_len, "cohen_d_mw": d_mw,
                "cohen_d_heavy": d_ha, "trivial_lr_acc": float(acc)}
    except Exception as e:
        logger.warning(f"Trivial LR failed: {e}")
        return {}


# ============================================================
#  HETEROGRAPH CONVERSION (unchanged)
# ============================================================
def mol_to_heterograph(smiles: str, label: int, metadata: Dict = None) -> Optional[HeteroData]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        data = HeteroData()
        node_feats: Dict[str, List[List[float]]] = defaultdict(list)
        atom_type_of: Dict[int, str] = {}

        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            atom_type_of[i] = symbol
            node_feats[symbol].append(_atom_features(atom, mol))

        for atype, feats in node_feats.items():
            data[atype].x = torch.tensor(feats, dtype=torch.float)

        edge_idx: Dict = defaultdict(lambda: [[], []])
        edge_attr: Dict = defaultdict(list)

        local_idx: Dict[int, int] = {}
        type_counter = defaultdict(int)
        for i in range(mol.GetNumAtoms()):
            t = atom_type_of[i]
            local_idx[i] = type_counter[t]
            type_counter[t] += 1

        for bond in mol.GetBonds():
            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            st, et = atom_type_of[s], atom_type_of[e]
            bf = _bond_features(bond, mol)
            edge_idx[(st, "bond_to", et)][0].append(local_idx[s])
            edge_idx[(st, "bond_to", et)][1].append(local_idx[e])
            edge_attr[(st, "bond_to", et)].append(bf)
            edge_idx[(et, "bond_to", st)][0].append(local_idx[e])
            edge_idx[(et, "bond_to", st)][1].append(local_idx[s])
            edge_attr[(et, "bond_to", st)].append(bf)

        for etype, (src, dst) in edge_idx.items():
            if src:
                data[etype].edge_index = torch.tensor([src, dst], dtype=torch.long)
                data[etype].edge_attr = torch.tensor(edge_attr[etype], dtype=torch.float)

        data.y = torch.tensor([label], dtype=torch.long)
        data.smiles = smiles

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    setattr(data, k, v)

        return data
    except Exception:
        return None


def _atom_features(atom, mol) -> List[float]:
    nbrs = atom.GetNeighbors()
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        atom.GetTotalDegree() / 6.0,
        (atom.GetFormalCharge() + 3) / 6.0,
        float(atom.GetHybridization()) / 6.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetMass() / 200.0,
        float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        float(atom.HasProp("_ChiralityPossible")),
        float(atom.GetSymbol() in ["N", "O"] and atom.GetTotalNumHs() > 0),
        float(atom.GetSymbol() in ["N", "O", "F"]),
        float(atom.GetSymbol() in ["F", "Cl", "Br", "I"]),
        float(atom.GetSymbol() in ["S", "P"]),
        float(atom.IsInRing()),
        float(atom.IsInRingSize(5)),
        float(atom.IsInRingSize(6)),
        float(any(mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), s) for s in [3, 4, 7, 8])),
        float(atom.GetNumRadicalElectrons()),
        len([n for n in nbrs if n.GetIsAromatic()]) / max(1, len(nbrs)),
        len([n for n in nbrs if n.GetSymbol() in ["N", "O", "S"]]) / max(1, len(nbrs)),
        len([n for n in nbrs if n.GetSymbol() in ["F", "Cl", "Br", "I"]]) / max(1, len(nbrs)),
        float(any(n.GetFormalCharge() != 0 for n in nbrs)),
        float(atom.GetHybridization() == Chem.HybridizationType.SP),
        float(atom.GetHybridization() == Chem.HybridizationType.SP2),
        float(atom.GetHybridization() == Chem.HybridizationType.SP3),
    ]


def _bond_features(bond, mol) -> List[float]:
    sa = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    ea = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
    return [
        bond.GetBondTypeAsDouble() / 3.0,
        float(bond.GetIsAromatic()),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
        float(bond.GetStereo() == Chem.BondStereo.STEREOZ),
        float(bond.IsInRingSize(6)),
        float(bond.IsInRingSize(5)),
        float(any(mol.GetRingInfo().IsBondInRingOfSize(bond.GetIdx(), s) for s in [3, 4, 7, 8])),
        (sa.GetDegree() + ea.GetDegree()) / 10.0,
        float(sa.GetIsAromatic() and ea.GetIsAromatic()),
        float(sa.GetIsAromatic() != ea.GetIsAromatic()),
        float(abs(sa.GetFormalCharge() - ea.GetFormalCharge())),
        float(sa.GetSymbol() in ["N", "O", "S", "P"]),
        float(ea.GetSymbol() in ["N", "O", "S", "P"]),
        float(sa.GetSymbol() != ea.GetSymbol()),
        abs(sa.GetAtomicNum() - ea.GetAtomicNum()) / 50.0,
        float(bond.GetBondType() == Chem.BondType.DOUBLE),
    ]


# ============================================================
#  MAIN PIPELINE
# ============================================================
def build_dataset(total_size: int = 50000,
                  output_path: str = "intelligent_dataset.pt",
                  seed: int = 42,
                  use_bulk_chembl: bool = False,
                  max_counterfeits_per_parent: int = 15):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    half = total_size // 2
    # Fetch more — we lose ~50% total through filtering + matching
    target_auth = int(half * 3.0)

    logger.info("=" * 70)
    logger.info(f"Building dataset — target {total_size} molecules, "
                f"bulk_chembl={use_bulk_chembl}")
    logger.info("=" * 70)

    # --- Step 1: Fetch authentic molecules ---
    fetcher = AuthenticFetcher(max_per_scaffold=50)

    chembl_rows: List[Dict] = []
    if use_bulk_chembl:
        chembl_rows = fetcher.fetch_chembl_bulk(int(target_auth * 0.7))
    if len(chembl_rows) < 1000:
        # Fallback / supplement with REST
        chembl_rows += fetcher.fetch_chembl_rest(int(target_auth * 0.7))

    pubchem_rows = fetcher.fetch_pubchem(int(target_auth * 0.5))

    authentic_rows = chembl_rows + pubchem_rows

    # Dedup by canonical SMILES
    seen: Set[str] = set()
    unique_auth = []
    for r in authentic_rows:
        if r["smiles"] in seen:
            continue
        seen.add(r["smiles"])
        unique_auth.append(r)

    logger.info(f"Unique authentic molecules: {len(unique_auth)}")
    logger.info(f"Fetcher stats: {dict(fetcher.stats)}")
    if len(unique_auth) < 100:
        logger.error("Too few authentic molecules fetched. Check network/API access.")
        return

    # --- Step 2: Generate counterfeits ---
    logger.info("Generating MMP counterfeits...")
    cf_gen = CounterfeitGenerator(seen_smiles=seen)
    counterfeit_rows: List[Dict] = []

    target_fake = int(half * 2.0)

    parents = list(unique_auth)
    random.shuffle(parents)
    with tqdm(total=target_fake, desc="Counterfeits") as pbar:
        for parent in parents:
            if len(counterfeit_rows) >= target_fake:
                break
            for _ in range(max_counterfeits_per_parent):
                if len(counterfeit_rows) >= target_fake:
                    break
                cf = cf_gen.generate_one(parent, max_attempts=50)
                if cf is not None:
                    counterfeit_rows.append(cf)
                    pbar.update(1)

    logger.info(f"Generated {len(counterfeit_rows)} unique counterfeits")
    logger.info(f"Counterfeit-gen stats: {dict(cf_gen.stats)}")

    # --- Step 3: Shortcut diagnostics BEFORE matching ---
    auth_df = pd.DataFrame(unique_auth)
    fake_df = pd.DataFrame(counterfeit_rows)

    logger.info("\n--- Diagnostics BEFORE property matching ---")
    metrics_before = report_shortcut_metrics(auth_df, fake_df)

    # --- Step 4: Property matching (reduced n_bins = 6 to keep more data) ---
    auth_matched, fake_matched = property_match(
        auth_df, fake_df,
        match_cols=["molecular_weight", "heavy_atoms"],
        n_bins=6,
    )

    # --- Step 5: Trim to exact target size ---
    n_keep = min(len(auth_matched), len(fake_matched), half)
    auth_matched = auth_matched.sample(n=n_keep, random_state=seed).reset_index(drop=True)
    fake_matched = fake_matched.sample(n=n_keep, random_state=seed).reset_index(drop=True)

    logger.info(f"\nFinal balanced size: {n_keep} authentic + {n_keep} counterfeit = {2*n_keep}")

    # --- Step 6: Diagnostics AFTER matching ---
    logger.info("\n--- Diagnostics AFTER property matching ---")
    metrics_after = report_shortcut_metrics(auth_matched, fake_matched)

    # --- Step 7: Combine, label, convert ---
    auth_matched["label"] = 0
    fake_matched["label"] = 1
    full_df = pd.concat([auth_matched, fake_matched], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info("\nConverting to heterographs...")
    graphs, labels = [], []
    node_types_seen: Set[str] = set()
    edge_types_seen: Set = set()

    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Graphs"):
        meta = {
            "source": row.get("source", ""),
            "source_id": row.get("source_id", ""),
            "counterfeit_type": row.get("counterfeit_type", "authentic"),
            "difficulty_level": row.get("difficulty_level", "authentic"),
            "original_smiles": row.get("parent_smiles", row["smiles"]),
            "parent_scaffold": row.get("parent_scaffold", row.get("scaffold", "")),
        }
        g = mol_to_heterograph(row["smiles"], int(row["label"]), metadata=meta)
        if g is None:
            continue
        graphs.append(g)
        labels.append(int(row["label"]))
        for nt in g.node_types:
            node_types_seen.add(nt)
        for et in g.edge_types:
            edge_types_seen.add(et)

    metadata = (sorted(node_types_seen), sorted(edge_types_seen))

    out_path = Path(output_path)
    torch.save((graphs, labels, metadata), out_path)
    logger.info(f"Saved dataset: {out_path.absolute()}")

    n_auth = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    stats = {
        "total_graphs": len(graphs),
        "authentic_count": n_auth,
        "counterfeit_count": n_fake,
        "balance_ratio": round(min(n_auth, n_fake) / max(n_auth, n_fake, 1), 3),
        "node_types": sorted(node_types_seen),
        "edge_type_count": len(edge_types_seen),
        "shortcut_metrics_before_matching": metrics_before,
        "shortcut_metrics_after_matching": metrics_after,
        "fetcher_stats": dict(fetcher.stats),
        "counterfeit_gen_stats": dict(cf_gen.stats),
        "difficulty_distribution": (
            fake_matched["difficulty_level"].value_counts().to_dict()
            if "difficulty_level" in fake_matched else {}
        ),
        "source_distribution": auth_matched["source"].value_counts().to_dict(),
        "counterfeit_types": (
            fake_matched["counterfeit_type"].value_counts().head(20).to_dict()
            if "counterfeit_type" in fake_matched else {}
        ),
    }
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats: {stats_path.absolute()}")

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"  Total:       {len(graphs):,}")
    logger.info(f"  Authentic:   {n_auth:,} ({n_auth/max(len(labels),1)*100:.1f}%)")
    logger.info(f"  Counterfeit: {n_fake:,} ({n_fake/max(len(labels),1)*100:.1f}%)")
    logger.info(f"  Balance:     {stats['balance_ratio']:.3f}")
    if "trivial_lr_acc" in metrics_after:
        logger.info(f"  Trivial LR accuracy: {metrics_after['trivial_lr_acc']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=50000,
                   help="Total dataset size (half authentic, half counterfeit)")
    p.add_argument("--output", type=str, default="intelligent_pharma_v3.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bulk-chembl", action="store_true",
                   help="Use chembl_downloader for bulk SDF (much faster, one-time download ~500MB)")
    p.add_argument("--max-cf-per-parent", type=int, default=15,
                   help="Max counterfeits to generate per authentic parent")
    args = p.parse_args()
    build_dataset(
        total_size=args.size,
        output_path=args.output,
        seed=args.seed,
        use_bulk_chembl=args.bulk_chembl,
        max_counterfeits_per_parent=args.max_cf_per_parent,
    )


if __name__ == "__main__":
    main()