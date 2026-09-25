set -e
cd /Users/ediionut23/Desktop/Licenta/Counterfeit-Detector
source venv/bin/activate
echo "=== [1/6] FETCH real molecules ($(date +%H:%M)) ==="
python fetch_public_molecules.py --target 70000 --out public_molecules_100k.smi
echo "=== [2/6] MINE catalogue ($(date +%H:%M)) ==="
python mine_mmp_rules.py --smiles-file public_molecules_100k.smi --workdir mmp_mining_100k --min-support 8 --out mined_transformations
echo "=== [3/6] Version M ~50k ($(date +%H:%M)) ==="
python build_version_m.py --parents public_molecules_100k.smi --per-category 10000 --out version_m
echo "=== [4/6] Version E ($(date +%H:%M)) ==="
python build_version_e.py --parents public_molecules_100k.smi --no-adversarial --max-parents 800 --per-parent-keep 12 --out version_e
echo "=== [5/6] ASSEMBLE ($(date +%H:%M)) ==="
python assemble_benchmark.py --authentic public_molecules_100k.smi --min-subcat 4 --out benchmark
echo "=== [6/6] CONFORMERS 3D ($(date +%H:%M)) ==="
python generate_conformers.py
echo "=== ALL DONE ($(date +%H:%M)) ==="
