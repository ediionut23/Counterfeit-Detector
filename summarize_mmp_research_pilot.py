"""Summarize the three-seed dedicated MMP challenge and training ablation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path("research_runs")
BASE = ROOT / "mmp_dedicated_1000"
HARD = ROOT / "mmp_dedicated_hard_train"
SEEDS = (42, 43, 44)
MODELS = (("properties", "3 properties"), ("rf", "Morgan + RF"),
          ("xgboost", "Morgan + XGBoost"))


def result_path(root: Path, seed: int, filename: str) -> Path:
    seed_path = root / f"seed{seed}" / filename
    return seed_path if seed_path.exists() else root / filename


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def main() -> None:
    unassigned = read_json(ROOT / "mmp_unassigned_cohort/summary.json")
    split = read_json(BASE / "summary.json")
    augmented = read_json(HARD / "summary.json")
    shared = read_json(ROOT / "shared_exact_edit_signatures.json")
    base = {seed: read_json(result_path(BASE, seed, "evaluation.json")) for seed in SEEDS}
    hard = {seed: read_json(result_path(HARD, seed, "evaluation.json")) for seed in SEEDS}
    changes = {seed: read_json(result_path(HARD, seed, "paired_ablation.json"))
               for seed in SEEDS}
    if len({result["challenge_sha256"] for result in (*base.values(), *hard.values())}) != 1:
        raise ValueError("Results do not share the same held-out challenge")
    with (BASE / "challenge.csv").open(newline="") as stream:
        positives = [row for row in csv.DictReader(stream) if row["label"] == "1"]
    categories = Counter(row["category"] for row in positives)
    positive_smiles = {row["smiles"] for row in positives}
    with Path("final_dataset/molecules.csv").open(newline="") as stream:
        dual_method_count = sum(row["smiles"] in positive_smiles and row["methods"] == "EM"
                                for row in csv.DictReader(stream))
    gaps = {name: statistics.mean(float(row[name]) for row in positives)
            for name in ("sim_gap", "mw_gap", "logp_gap", "heavy_gap")}

    lines = [
        "# Pilot: analogi moleculari generați și controale MMP apropiate", "",
        "## Protocol", "",
        f"Din banca finală au rezultat {fmt_int(unassigned['matched_parent_pairs'])} perechi candidate, "
        f"din care am rezervat {fmt_int(split['selected_pairs'])} pentru test: "
        f"{fmt_int(split['selected_by_method']['M'])} relații M și "
        f"{fmt_int(split['selected_by_method']['E'])} relații E. "
        "Fiecare pereche are un analog generat și o moleculă din biblioteca sursă, "
        "legate de același părinte. Cele trei partiții sunt separate la nivel de "
        "moleculă, componentă părinte–copil și schelet chimic al părintelui și al copilului.",
        "",
        f"Antrenare: {fmt_int(sum(split['split_label_counts']['train'].values()))} rânduri; "
        f"validare: {fmt_int(sum(split['split_label_counts']['val'].values()))}; "
        f"test: {fmt_int(sum(split['split_label_counts']['test'].values()))}. "
        "Pragul fiecărui model este ales pe validare pentru FPR ≤ 1%, "
        "fără ajustare pe test. Antrenarea folosește numai structura moleculei.",
        "",
        f"Potrivirea testului are diferență medie absolută de similitudine "
        f"{gaps['sim_gap']:.3f}, masă moleculară {gaps['mw_gap']:.1f} Da, "
        f"logP {gaps['logp_gap']:.3f} și {gaps['heavy_gap']:.2f} atomi grei. "
        f"Sunt {fmt_int(len(set(row['parent_scaffold'] for row in positives)))} schelete de părinte. "
        "Categoriile pozitive sunt: "
        + ", ".join(f"{name} {count}" for name, count in categories.most_common()) + ".",
        "",
        "## Rezultate la pragul calibrat", "",
        "Valorile sunt media ± deviația standard pentru semințele modelului 42, 43, 44; "
        "împărțirea datelor rămâne aceeași.", "",
        "| Model | Negative de antrenare | FPR validare | FPR controale MMP | TPR analogi generați | AUC test |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, label in MODELS:
        for root, title, results in ((BASE, "inițiale", base), (HARD, "+ MMP dificile", hard)):
            models = [results[seed]["models"][name] for seed in SEEDS]
            lines.append(f"| {label} | {title} | "
                         f"{fmt([m['validation_fpr'] for m in models])} | "
                         f"{fmt([m['challenge_fpr'] for m in models])} | "
                         f"{fmt([m['challenge_tpr'] for m in models])} | "
                         f"{fmt([m['challenge_auc'] for m in models])} |")
    lines += ["", "Pentru RF inițial, la sămânța 42: 42/1.000 alarme false pe "
              "controale MMP și 124/1.000 analogi detectați; intervalul bootstrap "
              "grupat după scheletul părintelui este "
              f"{base[42]['models']['rf']['challenge_fpr_95pct_scaffold_bootstrap_ci']} "
              "pentru FPR. AUC 0,678 în acest test descrie doar discriminarea "
              "provenienței sintetice față de biblioteca sursă.", "",
              "## Intervenția cu exemple dificile", "",
              f"Am înlocuit {augmented['replaced_training_negatives']} exemple negative "
              "de antrenare cu controale MMP apropiate de părinți din antrenare; "
              "mărimea antrenării, validarea și testul au rămas fixe. "
              "Diferențele FPR/TPR sunt calculate pe aceleași perechi, "
              "cu bootstrap grupat după schelet.", "",
              "| Model | Sămânță | Δ FPR test (IC 95%) | Δ TPR test (IC 95%) |",
              "|---|---:|---:|---:|",]
    for name, label in MODELS[1:]:
        for seed in SEEDS:
            data = changes[seed]["models"][name]
            fpr, tpr = data["fpr"], data["tpr"]
            lines.append(f"| {label} | {seed} | {fpr['change']:+.3f} "
                         f"({fpr['95pct_parent_scaffold_bootstrap_ci']}) | "
                         f"{tpr['change']:+.3f} "
                         f"({tpr['95pct_parent_scaffold_bootstrap_ci']}) |")
    lines += ["", "Pentru RF, intervalele schimbării FPR includ zero la toate "
              "cele trei semințe. Pentru XGBoost, sămânța 44 arată chiar o "
              "creștere a FPR. Înlocuirea a 130 de negative nu demonstrează "
              "o îmbunătățire robustă.", "",
              "## Interpretare și limite", "",
              "- Eticheta pozitivă indică generarea unui analog; eticheta negativă "
              "indică proveniența din biblioteca sursă. Niciuna nu certifică "
              "statutul unui produs farmaceutic sau efectul biologic.",
              "- Perechile MMP provin dintr-o bibliotecă folosită și la minarea "
              "regulilor M; acesta este un test intern, nu validare externă.",
              "- Relațiile M/E din test provin din categorii diferite, iar unele "
              f"structuri au fost generate de ambele metode ({dual_method_count} din "
              f"{fmt_int(len(positives))}). Diferențele M versus E "
              "nu izolează efectul generatorului.",
              "- Cele trei semințe testează variabilitatea modelului pe un singur "
              "split. Generalizarea cere mai multe splituri și controale dintr-o "
              "sursă independentă.",
              "- FPR pe test depășește în mod observabil FPR pe validare pentru "
              "RF și XGBoost; pragul calibrat pe sursa inițială nu garantează "
              "același număr de alarme pe analogi MMP.", "",
              "- Testul este echilibrat 1:1, iar intervalele bootstrap sunt "
              "condiționate de această selecție și gruparea după scheletul părintelui.", "",
              "## Fezabilitatea unui test M→E mai strict", "",
              f"Auditul relațiilor cu edit exact și structuri generate exclusiv printr-o "
              f"metodă găsește {shared['shared_signatures_with_100_children_per_method']} "
              "semnături elementare cu cel puțin 100 de structuri M și 100 E. "
              "Aceasta oferă straturi candidate în care tipul de schimbare poate "
              "fi ținut mai constant. O semnătură bazată pe numărul de elemente "
              "nu garantează aceeași transformare de graf sau același context chimic.", "",
              "| Schimbare elementară | Structuri M | Structuri E |",
              "|---|---:|---:|",]
    for row in shared["signatures"][:8]:
        lines.append(f"| {row['signature']} | {fmt_int(row['M_children'])} | "
                     f"{fmt_int(row['E_children'])} |")
    lines += ["", f"Fișierul complet este `research_runs/shared_exact_edit_signatures.json`; "
              f"manifestul cu {fmt_int(shared['counts']['accepted_relations'])} relații "
              "este `research_runs/shared_exact_edit_relations.csv`. "
              "Următorul test trebuie să verifice editul local, să excludă toate "
              "structurile EM și să păstreze aceleași familii în train și test.", "",
              "## Comenzi de reproducere", "",
              "```bash",
              "venv/bin/python build_mmp_hard_control_challenge.py --positive-relations final_dataset/relations.csv --unassigned-cohort --out research_runs/mmp_unassigned_cohort",
              "venv/bin/python construct_dedicated_mmp_split.py",
              "venv/bin/python augment_train_with_mmp_controls.py",
              "venv/bin/python audit_shared_exact_edit_signatures.py",
              "for root in research_runs/mmp_dedicated_1000 research_runs/mmp_dedicated_hard_train; do",
              "  for seed in 42 43 44; do",
              "    run_dir=\"$root/seed$seed\"",
              "    venv/bin/python train_baseline.py --train-csv \"$root/dataset.csv\" --train-only --target-fpr 0.01 --xgboost --jobs 2 --seed \"$seed\" --name mmp_pilot --out \"$run_dir/baseline.json\" --predictions-dir \"$run_dir/predictions\" --models-dir \"$run_dir/models\"",
              "    venv/bin/python evaluate_mmp_hard_control_challenge.py --dataset \"$root/dataset.csv\" --challenge \"$root/challenge.csv\" --models \"$run_dir/models\" --out \"$run_dir\"",
              "  done",
              "done",
              "for seed in 42 43 44; do",
              "  venv/bin/python compare_mmp_training_ablation.py --base \"research_runs/mmp_dedicated_1000/seed$seed\" --hard \"research_runs/mmp_dedicated_hard_train/seed$seed\"",
              "done",
              "venv/bin/python summarize_mmp_research_pilot.py",
              "```", "",
              "Următorul protocol: splituri repetate, M și E exclusiv pe clase chimice "
              "comparabile, controale dintr-o sursă independentă și comparații "
              "cu buget egal de selecție a hiperparametrilor.", "",
    ]
    destination = ROOT / "MMP_PILOT_REPORT.md"
    destination.write_text("\n".join(lines))
    print(destination)


if __name__ == "__main__":
    main()
