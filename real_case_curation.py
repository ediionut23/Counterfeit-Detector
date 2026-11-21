"""Manually curated primary-source findings, accessed 2026-10-31.

Each record is an alert/sample/batch-level report, not an independently
collected tablet or a measured spectrum. Compound IDs denote active moieties;
the original pharmaceutical salt/form is retained in the claim text.
"""

CURATION_DATE = "2026-10-31"

SOURCES = {
    "who_2019_02": {
        "authority": "WHO", "date": "2019-02-01", "title": "Medical Product Alert 2/2019: Falsified ICLUSIG",
        "url": "https://www.who.int/news/item/01-02-2019-medical-product-alert-n-2-2019-%28english-version%29",
        "locator": "Separate laboratory findings for batches 25A19E09 and PR072875",
        "excerpt": "does not contain Ponatinib and instead contains paracetamol",
    },
    "who_2017_04": {
        "authority": "WHO", "date": "2017-11-27", "title": "Medical Product Alert 4/2017: Falsified Penicillin V",
        "url": "https://www.who.int/news/item/27-11-2017-medical-product-alert-n-4-2017--falsified-penicillin-v",
        "annex_url": "https://cdn.who.int/media/docs/default-source/substandard-and-falsified/n4_2017-falsified-phenoxym-pencillin_en.pdf",
        "locator": "Investigation bullet on laboratory analysis; annex batch table",
        "excerpt": "do not contain any penicillin V (phenoxymethyl penicillin) but instead contain 50 mg of paracetamol",
    },
    "who_2019_06": {
        "authority": "WHO", "date": "2019-04-17", "title": "Medical Product Alert 6/2019: Falsified hydrochlorothiazide",
        "url": "https://www.who.int/news/item/17-04-2019-medical-product-alert-n-6-2019-%28english-version%29",
        "locator": "Confirmatory laboratory analysis; batch 16G04",
        "excerpt": "contains approximately 5mg of glibenclamide",
    },
    "who_2015_04": {
        "authority": "WHO", "date": "2015-07-02", "title": "Medical Product Alert 4/2015: Falsified Diazepam",
        "url": "https://www.who.int/news/item/02-07-2015-medical-product-alert-n-4-2015--adverse-reactions-caused-by-falsified-diazepam",
        "locator": "PRODUCT ONE has laboratory results; PRODUCT TWO explicitly untested",
        "excerpt": "does not contain Diazepam, but contains between 10mg to 20mg of Haloperidol per tablet",
    },
    "who_2018_02": {
        "authority": "WHO", "date": "2018-03-02", "title": "Medical Product Alert 2/2018: Falsified Augmentin",
        "url": "https://www.who.int/news/item/02-03-2018-medical-product-alert-n-2-2018--falsified--augmentin",
        "annex_url": "https://www.who.int/docs/default-source/essential-medicines/medical-alert-2019/n2-2018whoalert-falsified-augmentin.pdf",
        "locator": "Quality-assurance analysis bullet; annex batch/product table",
        "excerpt": "Quality assurance laboratory analysis did not identify any of the expected active ingredients",
    },
    "who_2021_03": {
        "authority": "WHO", "date": "2021-08-10", "title": "Medical Product Alert 3/2021: Falsified CYTOTEC",
        "url": "https://www.who.int/news/item/10-08-2021-medical-product-alert-n-3-2021",
        "annex_url": "https://cdn.who.int/media/docs/default-source/substandard-and-falsified/n3_2021-falsified-cytotec_en.pdf",
        "locator": "Separate B16519 and 14660 bullets; annex spells second batch B14660",
        "excerpt": "Laboratory analysis of samples has also confirmed the product does not contain any active ingredient",
    },
    "who_2022_02": {
        "authority": "WHO", "date": "2022-03-09", "title": "Medical Product Alert 2/2022: Falsified DESREM",
        "url": "https://www.who.int/news/item/09-03-2022-medical-product-alert-n-2-2022-falsified-desrem-%28remdesivir%29",
        "annex_url": "https://cdn.who.int/media/docs/default-source/substandard-and-falsified/n2_2022_falsified-desrem_en.pdf",
        "locator": "Manufacturer laboratory statement applies to listed products; annex batch table",
        "excerpt": "do not contain any of the stated active pharmaceutical ingredient (remdesivir)",
    },
    "who_2024_03": {
        "authority": "WHO", "date": "2024-08-05", "title": "Medical Product Alert 3/2024: Falsified oxymorphone hydrochloride",
        "url": "https://www.who.int/news/item/05-08-2024-medical-product-alert-n-3-2024--falsified-%28contaminated%29-oxymorphone-hydrochloride-40mg",
        "locator": "Alert Summary laboratory finding and manufacturer confirmation",
        "excerpt": "found that the tablets contained metonitazene instead",
    },
    "who_2024_04": {
        "authority": "WHO", "date": "2024-10-10", "title": "Medical Product Alert 4/2024: Falsified USP/EP propylene glycol",
        "url": "https://www.who.int/news/item/10-10-2024-medical-product-alert-n-4-2024--falsified-usp-ep-propylene-glycol",
        "locator": "Only batch F8900L8PPD6 has the reported contamination measurement",
        "excerpt": "contaminated with Ethylene Glycol at 0.7624% v/v",
    },
    "doj_2010_alli": {
        "authority": "US DOJ / FDA Forensic Chemistry Center", "date": "2010-03-25",
        "title": "Two arrested for illegally trafficking counterfeit weight loss medication",
        "url": "https://www.justice.gov/archive/usao/co/news/2010/March10/3_25_10.html",
        "annex_url": "https://www.justice.gov/archive/usao/co/news/2010/March10/Zhou_Complaint.pdf",
        "supporting_url": "https://www.justice.gov/archive/usao/co/news/2011/January2011/1_28_11.html",
        "locator": "FDA Forensic Chemistry Center consumer sample; affidavit p19 identifies claimed orlistat",
        "excerpt": "The pills contained Sibutramine and were determined to be fake versions of Alli",
    },
    "nsw_2024_diazepam": {
        "authority": "NSW Health / Forensic & Analytical Science Service", "date": "2024-05-16",
        "title": "Counterfeit diazepam tablets containing other drugs",
        "url": "https://www.health.nsw.gov.au/sabs/Documents/2024-si-003.PDF",
        "supporting_url": "https://www.health.nsw.gov.au/aod/public-drug-alerts/Pages/counterfeit-diazepam-may24.aspx",
        "locator": "Page 1 background, three distinct tested presentations/batches",
        "excerpt": "One batch was found to contain only bromazolam",
    },
}

# Resolve these exact names through PubChem; do not enter guessed SMILES.
COMPOUNDS = {
    "ponatinib": {"query": "ponatinib"},
    "paracetamol": {"query": "acetaminophen"},
    "phenoxymethylpenicillin": {"query": "phenoxymethylpenicillin"},
    "hydrochlorothiazide": {"query": "hydrochlorothiazide"},
    "glibenclamide": {"query": "glibenclamide"},
    "diazepam": {"query": "diazepam"},
    "haloperidol": {"query": "haloperidol"},
    "amoxicillin": {"query": "amoxicillin"},
    "clavulanic_acid": {"query": "clavulanic acid"},
    "misoprostol": {"query": "misoprostol", "mapping_note": "Clinical misoprostol is a stereoisomer mixture; one database record is not its measured composition."},
    "remdesivir": {"query": "remdesivir"},
    "oxymorphone": {"query": "oxymorphone"},
    "metonitazene": {"query": "metonitazene"},
    "orlistat": {"query": "orlistat"},
    "sibutramine": {"query": "sibutramine"},
    "bromazolam": {"query": "bromazolam"},
    "caffeine": {"query": "caffeine"},
    "propylene_glycol": {"query": "propylene glycol", "mapping_note": "Source does not resolve enantiomer composition."},
    "ethylene_glycol": {"query": "ethylene glycol"},
}


def make_case(case_id, source_id, product, declared, detected=(), absent=(),
              *, scenario="identified_api_substitution", batch=None, region="",
              claim_form="", strength="", analysis="reported_laboratory_analysis", notes=""):
    return {
        "case_id": case_id, "evidence_group_id": source_id, "source_id": source_id,
        "product_name": product, "batch_lot": batch, "reported_region": region,
        "declared_compound_ids": list(declared), "detected_compound_ids": list(detected),
        "not_detected_declared_compound_ids": list(absent),
        "declared_form_as_reported": claim_form, "declared_strength_as_reported": strength,
        "scenario": scenario, "analysis_status": analysis,
        "falsification_status": "authority_reported_falsification",
        "evidence_tier": "official_case_report", "partition": "external_case_challenge_only",
        "complete_sample_composition_known": False, "raw_measurements_available": False,
        "curation_notes": notes,
    }


CASES = [
    make_case("WHO2019_02_ICLUSIG_15", "who_2019_02", "ICLUSIG", ["ponatinib"], ["paracetamol"], ["ponatinib"], batch="25A19E09", region="Europe and Americas", claim_form="ponatinib hydrochloride", strength="15 mg", notes="Batch-specific laboratory finding; not an independent chemistry relation from the 45 mg batch."),
    make_case("WHO2019_02_ICLUSIG_45", "who_2019_02", "ICLUSIG", ["ponatinib"], ["paracetamol"], ["ponatinib"], batch="PR072875", region="Europe and Americas", claim_form="ponatinib hydrochloride", strength="45 mg", notes="Batch identifier alone is not proof of falsification for other packs/markets."),
    make_case("WHO2017_04_PENICILLIN", "who_2017_04", "Penicillin-V tablets", ["phenoxymethylpenicillin"], ["paracetamol"], ["phenoxymethylpenicillin"], batch="190", region="Cameroon", claim_form="phenoxymethylpenicillin", notes="Declared strength unreliable/misspelled in source; no strength inferred."),
    make_case("WHO2019_06_HCT", "who_2019_06", "Hydrochlorothiazide", ["hydrochlorothiazide"], ["glibenclamide"], ["hydrochlorothiazide"], batch="16G04", region="Cameroon", strength="50 mg", notes="Confirmatory laboratory analysis and manufacturer confirmation."),
    make_case("WHO2015_04_DIAZEPAM_ONE", "who_2015_04", "Diazepam / SOLINA bottle, AGOG-marked tablets", ["diazepam"], ["haloperidol"], ["diazepam"], region="Ituri, Democratic Republic of the Congo", strength="5 mg", notes="Laboratory evidence belongs only to PRODUCT ONE."),
    make_case("WHO2015_04_DIAZEPAM_TWO", "who_2015_04", "Diazepam / purported AGOG Pharma", ["diazepam"], region="Democratic Republic of the Congo", scenario="source_or_packaging_falsification", analysis="explicitly_not_laboratory_analysed", notes="Manufacturer confirmed false packaging; do not transfer PRODUCT ONE composition to this record."),
    make_case("WHO2018_02_AUGMENTIN", "who_2018_02", "Augmentin", ["amoxicillin", "clavulanic_acid"], absent=["amoxicillin", "clavulanic_acid"], scenario="declared_api_not_detected", batch="562626", region="Douala, Cameroon", claim_form="amoxicillin trihydrate + potassium clavulanate", strength="500 mg + 125 mg", notes="Neither declared API detected; substitute ingredients and full matrix unknown."),
    make_case("WHO2021_03_CYTOTEC_B16519", "who_2021_03", "CYTOTEC", ["misoprostol"], absent=["misoprostol"], scenario="declared_api_not_detected", batch="B16519", region="Cameroon, DRC, Ghana", strength="200 micrograms", notes="Source reports no active ingredient; does not identify excipients or substitutes."),
    make_case("WHO2021_03_CYTOTEC_B14660", "who_2021_03", "CYTOTEC", ["misoprostol"], scenario="source_or_packaging_falsification", batch="B14660", region="Cameroon and Nigeria", strength="200 micrograms", analysis="composition_not_reported", notes="Falsified expiry date; body says 14660 while annex says B14660. Composition unknown."),
    make_case("WHO2022_02_DESREM_GUATEMALA", "who_2022_02", "DESREM", ["remdesivir"], absent=["remdesivir"], scenario="declared_api_not_detected", batch="7605854B", region="Guatemala", strength="100 mg/vial", notes="Shared alert-level manufacturer assay statement; no batch-specific quantitative raw data."),
    make_case("WHO2022_02_DESREM_INDIA", "who_2022_02", "DESREM", ["remdesivir"], absent=["remdesivir"], scenario="declared_api_not_detected", batch="CRM21001MA", region="India", strength="100 mg/vial", notes="Same alert evidence family as Guatemala; do not count as independent chemistry."),
    make_case("WHO2024_03_OXYMORPHONE", "who_2024_03", "Oxymorphone Hydrochloride", ["oxymorphone"], ["metonitazene"], region="Finland", claim_form="oxymorphone hydrochloride", strength="40 mg", notes="Source says metonitazene instead; no numerical oxymorphone absence assay reported. Claimed 40 mg presentation was not made by genuine manufacturer."),
    make_case("WHO2024_04_PROPYLENE_GLYCOL", "who_2024_04", "DOW USP/EP Propylene Glycol", ["propylene_glycol"], ["ethylene_glycol"], scenario="identified_undeclared_contaminant", batch="F8900L8PPD6", region="Pakistan", notes="Measured contamination applies only to this batch. Propylene glycol presence/absence is not resolved; detected contaminant is not the whole sample."),
    make_case("DOJ2010_ALLI", "doj_2010_alli", "Alli", ["orlistat"], ["sibutramine"], ["orlistat"], region="United States", strength="60 mg capsules", notes="FDA laboratory consumer-sample finding reported by DOJ; affidavit identifies expected API. This is a sample/event aggregate, not a disclosed batch."),
    make_case("NSW2024_ANTENEX_BROMAZOLAM", "nsw_2024_diazepam", "Counterfeit Antenex 5 appearance", ["diazepam"], ["bromazolam"], ["diazepam"], region="New South Wales, Australia", strength="5 mg", notes="Off-white DM|5 / G; one tested batch, no public batch identifier. Only detected drug described, not excipient composition."),
    make_case("NSW2024_ANTENEX_MIXTURE", "nsw_2024_diazepam", "Counterfeit Antenex 5 appearance", ["diazepam"], ["paracetamol", "caffeine"], ["diazepam"], scenario="identified_substitute_mixture", region="New South Wales, Australia", strength="5 mg", notes="Another tested batch. Paracetamol and caffeine coexist; one mixture observation, never two counterfeit samples."),
    make_case("NSW2024_DAN5619_BROMAZOLAM", "nsw_2024_diazepam", "Diazepam overseas-product lookalike", ["diazepam"], ["bromazolam"], ["diazepam"], region="New South Wales, Australia", notes="Yellow DAN|5619 / 5. No manufacturer assigned; shares a substitution relation with the Antenex case."),
]

# Preserve reported quantitative observations without inventing assay limits,
# uncertainty intervals, full compositions, or salt-to-moiety conversions.
REPORTED_QUANTITIES = {
    "WHO2017_04_PENICILLIN": [{"compound_id": "paracetamol", "value": 50, "unit": "mg/tablet", "qualifier": "reported_value"}],
    "WHO2019_06_HCT": [{"compound_id": "glibenclamide", "value": 5, "unit": "mg/tablet", "qualifier": "approximately"}],
    "WHO2015_04_DIAZEPAM_ONE": [{"compound_id": "haloperidol", "minimum": 10, "maximum": 20, "unit": "mg/tablet", "qualifier": "reported_range"}],
    "WHO2024_04_PROPYLENE_GLYCOL": [{"compound_id": "ethylene_glycol", "value": 0.7624, "unit": "percent_v_v", "qualifier": "reported_value"}],
}
for _case in CASES:
    _case["reported_detected_quantities"] = REPORTED_QUANTITIES.get(_case["case_id"], [])
    if _case["source_id"] == "nsw_2024_diazepam":
        _case["curation_notes"] += " API-absence annotation follows the report's exclusive qualitative drug finding; no assay detection limit is supplied."
