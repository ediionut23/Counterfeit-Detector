"""Write a concise Romanian results table from the audited final bank."""

from __future__ import annotations

import json
from pathlib import Path


def fmt(value):
    return f"{value:,}".replace(",", ".") if isinstance(value, int) else str(value)


def main() -> None:
    root = Path("final_dataset")
    summary = json.loads((root / "summary.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    index = json.loads((root / "subcategories/index.json").read_text())
    conformers = root / "conformers_report.json"
    lines = [
        "# Rezultatele băncii finale de molecule contrafăcute",
        "",
        "Datele provin din `summary.json` și `audit.json`. Fiecare moleculă apare o singură dată în `molecules.csv`; o moleculă poate avea mai multe relații și poate figura în mai multe subcategorii.",
        "",
        "| Indicator | Valoare |",
        "|---|---:|",
        f"| Contrafaceri sintetice unice | {fmt(summary['n_unique_counterfeits'])} |",
        f"| Relații părinte–analog | {fmt(summary['n_distinct_relations'])} |",
        f"| Molecule autentice de referință | {fmt(summary['n_authentic_reference'])} |",
        f"| Molecule autentice eligibile pentru matching | {fmt(summary['n_authentic_eligible_for_training'])} |",
        f"| Subcategorii | {fmt(summary['n_subcategories'])} |",
        f"| Schele Bemis–Murcko ale analogilor | {fmt(summary['n_unique_child_scaffolds'])} |",
        f"| Audit structural independent | {'trecut' if audit['passed'] else 'EȘUAT'} |",
        "",
        "## Metode de generare",
        "",
        "| Metode prezente per moleculă | Molecule unice |",
        "|---|---:|",
    ]
    for method, count in summary["method_membership"].items():
        lines.append(f"| {method} | {fmt(count)} |")
    lines += [
        "",
        "`M` = reguli MMP minate; `E` = căutare evolutivă. `EM` înseamnă că aceeași structură a fost obținută prin ambele metode.",
        "",
        "## Calitate chimică și măști de editare",
        "",
        "| Măsură | Media | Mediana | P05 | P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("QED analog", "molecule_qed_distribution"),
        ("SA analog", "molecule_sa_distribution"),
        ("Tanimoto analog–părinte", "relation_similarity_distribution"),
        ("Atomi în mască nenulă", "nonmissing_edit_mask_atom_count_distribution"),
    ):
        stats = summary[key]
        lines.append(f"| {label} | {stats['mean']} | {stats['median']} | {stats['p05']} | {stats['p95']} |")
    lines += ["", "| Statutul măștii | Relații |", "|---|---:|"]
    for status, count in summary["edit_status_relation_counts"].items():
        lines.append(f"| {status} | {fmt(count)} |")
    lines += [
        "",
        "Măștile `exact_single_atom` permit evaluarea exactă a localizării editării. Măștile `mcs_approximate` și `source_approximate` sunt estimări și se evaluează separat.",
        "",
        "## Subcategorii selectabile",
        "",
        "| Metodă / categorie / subcategorie | Molecule unice în modul |",
        "|---|---:|",
    ]
    for name, meta in sorted(index.items()):
        lines.append(f"| {name} | {fmt(meta['unique_molecules'])} |")
    lines += [
        "",
        "Suma acestui tabel poate depăși numărul global de molecule deoarece modulele se pot suprapune.",
        "",
        "## Conformeri 3D",
        "",
    ]
    if conformers.exists():
        report = json.loads(conformers.read_text())
        lines += [
            f"Acoperire: {fmt(report['covered_total'])}/{fmt(report['wanted_unique_smiles'])} ({report['coverage']:.4%}).",
            f"Molecule rămase fără conformer: {fmt(report['missing'])}.",
        ]
    else:
        lines.append("Generarea 3D este în curs; `conformers_report.json` va conține acoperirea finală.")
    lines += [
        "",
        "## Interpretare",
        "",
        "Eticheta `1` identifică un analog sintetic generat relativ la un părinte din biblioteca de referință; nu demonstrează existența unui medicament contrafăcut comercial. Bibliotecile publice ChEMBL/PubChem folosite ca sursă nu certifică automat statutul de medicament aprobat al fiecărui părinte. Filtrele QED, SA, PAINS/Brenk și similaritate măsoară plauzibilitate computațională, nu sinteză sau activitate validate experimental.",
        "",
        "Pentru a crea un CSV de antrenare cu subcategoriile alese, vezi comenzile din `README.md`. Numărul de molecule autentice eligibile este mai mic decât numărul de analogi, astfel că banca modulară este sursa pentru mai multe experimente controlate, nu un singur set echilibrat 1:1.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
