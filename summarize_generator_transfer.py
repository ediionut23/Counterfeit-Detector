"""Write a reproducible summary of the two generator-transfer sensitivity runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_generator_transfer_split import parent_similarity


ROOT = Path("research_runs")
RUNS = {
    "descriptor_match": ROOT / "generator_transfer_exact",
    "parent_match": ROOT / "generator_transfer_parent_match",
    "local_match": ROOT / "generator_transfer_local_match",
}
DISPLAY = {"descriptor_match": "Proprietăți", "parent_match": "Părinți",
           "local_match": "Vecinătate locală"}
SEEDS = (42, 43, 44)
SPLIT_SEEDS = (20260927, 20260928, 20260929)
MODELS = {"properties": "3 proprietăți", "rf": "Morgan + RF",
          "xgboost": "Morgan + XGBoost"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(values: list[float], digits: int = 3) -> str:
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def main() -> None:
    runs = {}
    for label, root in RUNS.items():
        summary = json.loads((root / "summary.json").read_text())
        evaluations = {}
        for seed in SEEDS:
            result = json.loads((root / f"seed{seed}" / "evaluation.json").read_text())
            if result["dataset_sha256"] != summary["dataset_sha256"]:
                raise ValueError(f"Evaluation uses a different split: {root}, {seed}")
            evaluations[seed] = result
        pairs = defaultdict(dict)
        for row in read_csv(root / "dataset.csv"):
            if row["comparison_pair_id"]:
                pairs[row["comparison_pair_id"]][row["generation_method"]] = row
        if len(pairs) != summary["matched_test_pairs"] or any(
            set(pair) != {"M", "E"} for pair in pairs.values()
        ):
            raise ValueError(f"Malformed pairs in {root}")
        pairs = list(pairs.values())
        quality = {
            "parent_tanimoto_mean": float(np.mean([
                parent_similarity(pair["M"], pair["E"]) for pair in pairs])),
            "same_parent_scaffold": sum(
                pair["M"]["scaffold"] == pair["E"]["scaffold"] for pair in pairs),
            **{f"{column}_mean_absolute_gap": float(np.mean([
                abs(float(pair["M"][column]) - float(pair["E"][column]))
                for pair in pairs])) for column in ("mw", "logp", "heavy_atoms")},
        }
        runs[label] = (root, summary, evaluations, quality)

    lines = [
        "# Pilot: transferul M→E la editări elementare comparabile",
        "",
        "## Întrebarea și protocolul",
        "",
        "Poate un detector antrenat exclusiv pe analogi generați prin reguli MMP "
        "(`M`) să recunoască analogi produși prin căutare evolutivă (`E`), "
        "când tipul editării locale este comparabil? Acesta este un test intern "
        "de proveniență a structurilor virtuale; etichetele nu indică falsuri "
        "farmaceutice confirmate sau activitate biologică.",
        "",
        "Au fost selectate opt semnături de substituție cu un singur atom "
        "verificat: Cl→F, C→Cl, C→F, C→N, C→O, F→O, N→O și O→C. "
        "Structurile pozitive din antrenare și validare provin numai din `M`. "
        "Testul conține câte 40 de structuri exclusive `M` și `E` per semnătură, "
        "în 320 de perechi potrivite după proprietăți și similitudine față de "
        "părinte. Fiecare comparație AUC folosește aceleași 1.000 de molecule "
        "negative din biblioteca sursă. Spliturile nu suprapun structuri, "
        "componente părinte–copil sau scheletele părinților și copiilor. "
        "Pragul se alege numai pe negativele de validare pentru FPR ≤ 1%.",
        "",
        "Am repetat modelarea cu semințele 42, 43 și 44 pe fiecare split fix. "
        "A doua versiune a splitului penalizează suplimentar diferența de "
        "amprentă Morgan între părinții perechilor M/E. A treia cere "
        "identitatea primei vecinătăți a atomului substituit. Aceste "
        "variante sunt analize de sensibilitate a potrivirii, nu "
        "replici independente.",
        "",
        "## Compoziție și calitatea potrivirii",
        "",
        "| Variantă | Train M / negative | Val M / negative | Test M / E / negative | Tanimoto părinți, medie | Același schelet părinte | ΔMW, Da | ΔlogP | Δatomi grei |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (_, summary, _, quality) in runs.items():
        counts = summary["counts"]
        display = DISPLAY[label]
        lines.append(
            f"| {display} | {counts['train']['M_positive']} / {counts['train']['source_negative']} "
            f"| {counts['val']['M_positive']} / {counts['val']['source_negative']} "
            f"| {counts['test']['M_positive']} / {counts['test']['E_positive']} / {counts['test']['source_negative']} "
            f"| {quality['parent_tanimoto_mean']:.3f} "
            f"| {quality['same_parent_scaffold']}/{summary['matched_test_pairs']} "
            f"| {quality['mw_mean_absolute_gap']:.2f} "
            f"| {quality['logp_mean_absolute_gap']:.3f} "
            f"| {quality['heavy_atoms_mean_absolute_gap']:.3f} |"
        )
    lines += [
        "",
        "Constrângerile suplimentare de potrivire modifică și diferențele "
        "descriptorilor moleculelor copil. Nici una dintre variante nu "
        "echivalează complet întregul context chimic.",
        "",
        "## Rezultate",
        "",
        "FPR și TPR sunt la pragul de validare, iar AUC este calculat pentru "
        "fiecare grup pozitiv în raport cu **același** grup negativ din test. "
        "Valorile sunt medie ± deviație standard pentru cele trei semințe "
        "ale modelului; splitul rămâne fix.",
        "",
        "| Potrivire | Model | FPR test | TPR M | TPR E | AUC M | AUC E | ΔAUC E−M |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (_, _, evaluations, _) in runs.items():
        display = DISPLAY[label]
        for model, model_name in MODELS.items():
            values = [evaluations[seed]["models"][model] for seed in SEEDS]
            lines.append(
                f"| {display} | {model_name} "
                f"| {fmt([v['test_fpr'] for v in values])} "
                f"| {fmt([v['positive_method']['M']['tpr'] for v in values])} "
                f"| {fmt([v['positive_method']['E']['tpr'] for v in values])} "
                f"| {fmt([v['positive_method']['M']['auc_against_same_source_negatives'] for v in values])} "
                f"| {fmt([v['positive_method']['E']['auc_against_same_source_negatives'] for v in values])} "
                f"| {fmt([v['paired_E_minus_M_auc'] for v in values])} |"
            )
    lines += [
        "",
        "La sămânța 42, bootstrapul pe cele 320 de perechi fixe oferă "
        "următoarele intervale descriptive de 95% pentru ΔAUC E−M și ΔTPR "
        "E−M. Negativele testului rămân fixe în bootstrap.",
        "",
        "| Potrivire | Model | ΔAUC E−M [IC 95%] | ΔTPR E−M [IC 95%] |",
        "|---|---|---:|---:|",
    ]
    for label, (_, _, evaluations, _) in runs.items():
        display = DISPLAY[label]
        for model in ("rf", "xgboost"):
            value = evaluations[42]["models"][model]
            auc_ci = value["paired_E_minus_M_auc_95pct_pair_bootstrap_ci"]
            tpr_ci = value["paired_E_minus_M_tpr_95pct_pair_bootstrap_ci"]
            lines.append(
                f"| {display} | {MODELS[model]} "
                f"| {value['paired_E_minus_M_auc']:+.4f} [{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}] "
                f"| {value['paired_E_minus_M_tpr']:+.4f} [{tpr_ci[0]:+.4f}, {tpr_ci[1]:+.4f}] |"
            )
    lines += [
        "",
        "Același bootstrap, grupat după toate scheletele părinte/copil "
        "comune din test, produce următoarele intervale pentru ΔAUC "
        "la sămânța 42. Această grupare leagă și perechi diferite care "
        "împart un schelet, păstrând fix grupul de negative.",
        "",
        "| Potrivire | Grupuri de schelete | Cel mai mare grup | Model | ΔAUC E−M [IC 95% grupat] |",
        "|---|---:|---:|---|---:|",
    ]
    for label, (_, _, evaluations, _) in runs.items():
        display = DISPLAY[label]
        evaluation = evaluations[42]
        for model in ("rf", "xgboost"):
            value = evaluation["models"][model]
            ci = value["paired_E_minus_M_auc_95pct_scaffold_cluster_bootstrap_ci"]
            lines.append(
                f"| {display} | {evaluation['test_pair_scaffold_clusters']} "
                f"| {evaluation['largest_test_pair_scaffold_cluster']} "
                f"| {MODELS[model]} | {value['paired_E_minus_M_auc']:+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )
    lines += [
        "",
        "## Alte selecții ale perechilor și spliturilor",
        "",
        "Am repetat versiunea cu părinți potriviți folosind trei semințe "
        "de selecție. În această verificare, sămânța modelului este fixă "
        "la 42. Se schimbă perechile din test, negativele și partițiile; "
        "selecțiile pot totuși împărți molecule între ele.",
        "",
        "| Sămânță split | Train M | Model | FPR test | AUC M | AUC E | ΔAUC E−M [IC 95% grupat] |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for split_seed in SPLIT_SEEDS:
        root = (RUNS["parent_match"] if split_seed == SPLIT_SEEDS[0]
                else ROOT / f"generator_transfer_parent_seed{split_seed}")
        summary = json.loads((root / "summary.json").read_text())
        evaluation = json.loads((root / "seed42/evaluation.json").read_text())
        if summary["dataset_sha256"] != evaluation["dataset_sha256"]:
            raise ValueError(f"Evaluation uses a different split: {root}")
        for model in ("rf", "xgboost"):
            value = evaluation["models"][model]
            ci = value["paired_E_minus_M_auc_95pct_scaffold_cluster_bootstrap_ci"]
            lines.append(
                f"| {split_seed} | {summary['counts']['train']['M_positive']} "
                f"| {MODELS[model]} | {value['test_fpr']:.3f} "
                f"| {value['positive_method']['M']['auc_against_same_source_negatives']:.3f} "
                f"| {value['positive_method']['E']['auc_against_same_source_negatives']:.3f} "
                f"| {value['paired_E_minus_M_auc']:+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )
    lines += [
        "",
        "Direcția diferenței AUC persistă în aceste trei selecții interne. "
        "Numărul mic de selecții, suprapunerea posibilă între ele și "
        "proveniența comună a datelor limitează orice afirmație despre "
        "generalizare externă.",
        "",
        "Diferența AUC pe perechile potrivite numai după semnătura "
        "elementară este stabilă în acest pilot, dar nu izolează "
        "generatorul. Am verificat separat vecinătatea exactă a atomului "
        "modificat, mai jos. Diferența de TPR la un singur prag este mai "
        "puțin stabilă. FPR observat pe test ajunge la aproximativ 1–3%, "
        "deși pragul a fost ales pentru cel mult 1% pe validare; "
        "calibrările nu se transferă perfect nici între subseturile de "
        "molecule sursă.",
        "",
        "## Auditul vecinătății atomului editat",
        "",
        "Pentru fiecare relație pozitivă din test am refăcut substituția "
        "în sens invers și am verificat unicitatea poziției și masca "
        "atomilor editați. Toate cele 640 de relații din fiecare split "
        "trec acest audit. Două molecule dintr-o pereche au «aceeași "
        "vecinătate» când atomul substituit are același statut aromatic, "
        "aceeași apartenență la inel, același grad și aceleași tipuri "
        "de atomi vecini și legături în primul strat. Această condiție "
        "nu egalizează întregul schelet sau proprietățile biologice.",
        "",
        "| Split | Perechi cu vecinătate identică | Model | ΔAUC toate | ΔAUC vecinătate identică [IC 95% grupat] | ΔAUC vecinătate diferită |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    local_roots = {
        20260927: RUNS["parent_match"],
        20260928: ROOT / "generator_transfer_parent_seed20260928",
        20260929: ROOT / "generator_transfer_parent_seed20260929",
    }
    for split_seed, root in local_roots.items():
        local = json.loads((root / "local_context_summary.json").read_text())
        main_evaluation = json.loads((root / "seed42/evaluation.json").read_text())
        if local["pairs"] != main_evaluation["test_M"]:
            raise ValueError(f"Local audit has a different test size: {root}")
        for model in ("rf", "xgboost"):
            matched = local["models"]["42"][model]["same_first_shell"]
            unmatched = local["models"]["42"][model]["different_first_shell"]
            ci = matched["scaffold_cluster_bootstrap_95pct_ci"]
            lines.append(
                f"| {split_seed} | {local['same_first_shell_pairs']}/{local['pairs']} "
                f"| {MODELS[model]} | {main_evaluation['models'][model]['paired_E_minus_M_auc']:+.4f} "
                f"| {matched['paired_E_minus_M_auc']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] "
                f"| {unmatched['paired_E_minus_M_auc']:+.4f} |"
            )
    exact_local = json.loads((RUNS["descriptor_match"] / "local_context_summary.json").read_text())
    exact_eval = json.loads((RUNS["descriptor_match"] / "seed42/evaluation.json").read_text())
    lines += ["", "Și în splitul potrivit doar după proprietăți, "
              f"{exact_local['same_first_shell_pairs']}/{exact_local['pairs']} perechi au "
              "aceeași primă vecinătate. La sămânța 42, diferența AUC "
              f"scade de la {exact_eval['models']['rf']['paired_E_minus_M_auc']:+.4f} "
              f"la {exact_local['models']['42']['rf']['same_first_shell']['paired_E_minus_M_auc']:+.4f} "
              "pentru RF și de la "
              f"{exact_eval['models']['xgboost']['paired_E_minus_M_auc']:+.4f} "
              f"la {exact_local['models']['42']['xgboost']['same_first_shell']['paired_E_minus_M_auc']:+.4f} "
              "pentru XGBoost. Pe cele trei semințe de model ale celor două "
              "splituri principale, acest contrast păstrează aceeași "
              "direcție. Diferența mare agregată este concentrată în "
              "perechile cu vecinătăți locale diferite. Selecția condițională "
              "poate schimba și alte caracteristici, deci rezultatul "
              "nu dovedește că vecinătatea este singura cauză.", ""]
    explicit = runs["local_match"][2]
    explicit_local = json.loads((RUNS["local_match"] / "local_context_summary.json").read_text())
    if explicit_local["same_first_shell_pairs"] != explicit_local["pairs"]:
        raise ValueError("Explicit local-context split contains unmatched test pairs")
    lines += [
        "### Split construit cu vecinătate identică de la început",
        "",
        "Am generat încă un split cu 40 de perechi per semnătură și "
        "vecinătate identică impusă la selecție, înainte de antrenare. "
        "Auditul independent confirmă 320/320 perechi și 640/640 "
        "substituții inversabile cu măști corecte. Acest split folosește "
        "alte molecule, alte negative și alt set de antrenare decât cele "
        "de mai sus; comparăm direcția, nu o diferență cauzală între splituri.",
        "",
        "| Model | FPR test | AUC M | AUC E | ΔAUC E−M | TPR M | TPR E |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("rf", "xgboost"):
        values = [explicit[seed]["models"][model] for seed in SEEDS]
        lines.append(
            f"| {MODELS[model]} "
            f"| {fmt([v['test_fpr'] for v in values])} "
            f"| {fmt([v['positive_method']['M']['auc_against_same_source_negatives'] for v in values])} "
            f"| {fmt([v['positive_method']['E']['auc_against_same_source_negatives'] for v in values])} "
            f"| {fmt([v['paired_E_minus_M_auc'] for v in values], 4)} "
            f"| {fmt([v['positive_method']['M']['tpr'] for v in values])} "
            f"| {fmt([v['positive_method']['E']['tpr'] for v in values])} |"
        )
    lines += [
        "",
        "Când prima vecinătate este impusă explicit, diferența AUC "
        "rămâne negativă, dar este mai mică decât în splitul cu părinți "
        "potriviți fără această condiție. Nici potrivirea locală nu "
        "controlează întregul graf molecular ori proveniența bibliotecii.",
        "",
    ]
    lines += [
        "## Diferențe între semnături",
        "",
        "Pentru varianta cu părinți potriviți mai strâns, TPR-ul XGBoost la "
        "pragul de validare variază mult între editări. Fiecare celulă "
        "rezumă 40 de molecule per metodă și aceleași trei semințe de model; "
        "aceasta este o descriere a splitului, nu o comparație inferențială "
        "între familii.",
        "",
        "| Substituție | TPR M | TPR E | Δ E−M |",
        "|---|---:|---:|---:|",
    ]
    parent_evaluations = runs["parent_match"][2]
    for signature in runs["parent_match"][1]["signatures"]:
        by_seed = [parent_evaluations[seed]["models"]["xgboost"]["by_signature"][signature]
                   for seed in SEEDS]
        mean_m = float(np.mean([row["M"]["tpr"] for row in by_seed]))
        mean_e = float(np.mean([row["E"]["tpr"] for row in by_seed]))
        lines.append(f"| {signature} | {mean_m:.3f} | {mean_e:.3f} | {mean_e - mean_m:+.3f} |")
    lines += [
        "",
        "Media globală maschează această eterogenitate: de exemplu, "
        "C→F nu scade în varianta cu părinți potriviți, în timp ce O→C "
        "scade mult. Aceasta cere auditul contextului local și mai multe "
        "splituri înainte de a formula o explicație chimică.",
        "",
        "## Ce nu demonstrează încă pilotul",
        "",
        "- Perechile din primele două splituri au aceeași schimbare "
        "elementară, dar pot avea vecinătăți și schelete diferite. Al "
        "treilea split fixează prima vecinătate, însă contextul chimic "
        "mai îndepărtat rămâne variabil. Diferența AUC nu poate fi "
        "atribuită cauzal algoritmului de generare.",
        "- Toate selecțiile folosesc aceeași bancă sintetică și aceeași "
        "bibliotecă sursă. Cele trei semințe de model măsoară variabilitatea "
        "antrenării pe split fix; cele trei semințe de split sunt doar o "
        "primă verificare a selecției interne. Bootstrapul, inclusiv "
        "varianta grupată pe schelet, nu include variația unei surse noi.",
        "- Negativele sunt molecule catalogate în sursă, nu medicamente "
        "confirmate clinic. Rezultatele măsoară discriminarea provenienței "
        "datasetului, nu identificarea falsurilor reale.",
        "- TPR la prag mic este modest, iar FPR pe test depășește adesea "
        "bugetul de 1%; AUC nu rezolvă această limită operațională.",
        "",
        "## Pași de cercetare care urmează din rezultat",
        "",
        "1. Extinderea spliturilor repetate și evaluarea variației între "
        "ele; raportarea AUC și a calibrării per semnătură.",
        "2. Extinderea potrivirii de la prima vecinătate la două legături "
        "și evaluarea inversă `E→M` cu aceeași politică de selecție.",
        "3. Martori MMP și apoi controale dintr-o sursă independentă; "
        "recalibrare evaluată numai când există un set de validare extern.",
        "4. Abia după aceste controale, compararea GNN-urilor și a unei "
        "metode care reduce diferența de transfer fără să crească FPR.",
        "",
        "## Fișiere și reproducere",
        "",
        "Fiecare director păstrează `dataset.csv`, `matched_M_E_test.csv`, "
        "`summary.json`, rezultatele per sămânță și predicțiile per moleculă. "
        "Manifestele conțin SHA-256 pentru CSV-uri. Evaluatorul reface "
        "auditul de proveniență, separarea componentelor și scheletelor "
        "înainte de scorare. Auditul vecinătății produce "
        "`local_context_pairs.csv` și `local_context_summary.json`.",
        "",
        "```bash",
        "venv/bin/python audit_shared_exact_edit_signatures.py",
        "venv/bin/python build_generator_transfer_split.py --out research_runs/generator_transfer_exact",
        "venv/bin/python build_generator_transfer_split.py --parent-fingerprint-weight 3 --out research_runs/generator_transfer_parent_match",
        "venv/bin/python build_generator_transfer_split.py --parent-fingerprint-weight 3 --require-matched-local-shell --out research_runs/generator_transfer_local_match",
        "for split_seed in 20260928 20260929; do",
        "  venv/bin/python build_generator_transfer_split.py --seed \"$split_seed\" --parent-fingerprint-weight 3 --out \"research_runs/generator_transfer_parent_seed$split_seed\"",
        "done",
        "for root in research_runs/generator_transfer_exact research_runs/generator_transfer_parent_match research_runs/generator_transfer_local_match; do",
        "  for seed in 42 43 44; do",
        "    run_dir=\"$root/seed$seed\"",
        "    venv/bin/python train_baseline.py --train-csv \"$root/dataset.csv\" --train-only --target-fpr 0.01 --xgboost --jobs 2 --seed \"$seed\" --name \"$(basename \"$root\")\" --out \"$run_dir/baseline.json\" --predictions-dir \"$run_dir/predictions\" --models-dir \"$run_dir/models\"",
        "    venv/bin/python evaluate_generator_transfer.py --root \"$root\" --seed \"$seed\" > \"$run_dir/evaluation_stdout.json\"",
        "  done",
        "done",
        "for split_seed in 20260928 20260929; do",
        "  root=\"research_runs/generator_transfer_parent_seed$split_seed\"",
        "  run_dir=\"$root/seed42\"",
        "  mkdir -p \"$run_dir\"",
        "  venv/bin/python train_baseline.py --train-csv \"$root/dataset.csv\" --train-only --target-fpr 0.01 --xgboost --jobs 2 --seed 42 --name \"$(basename \"$root\")\" --out \"$run_dir/baseline.json\" --predictions-dir \"$run_dir/predictions\" --models-dir \"$run_dir/models\"",
        "  venv/bin/python evaluate_generator_transfer.py --root \"$root\" --seed 42 > \"$run_dir/evaluation_stdout.json\"",
        "done",
        "venv/bin/python audit_generator_local_context.py --root research_runs/generator_transfer_exact --seeds 42 43 44",
        "venv/bin/python audit_generator_local_context.py --root research_runs/generator_transfer_parent_match --seeds 42 43 44",
        "venv/bin/python audit_generator_local_context.py --root research_runs/generator_transfer_local_match --seeds 42 43 44",
        "for split_seed in 20260928 20260929; do",
        "  venv/bin/python audit_generator_local_context.py --root \"research_runs/generator_transfer_parent_seed$split_seed\" --seeds 42",
        "done",
        "venv/bin/python summarize_generator_transfer.py",
        "```",
        "",
    ]
    destination = ROOT / "GENERATOR_TRANSFER_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
