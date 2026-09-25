set -e
echo "=== [1/4] build_version_m ==="
python build_version_m.py --rules mined_transformations.py --parents public_molecules.smi --per-category 2000 --out version_m
echo "=== [2/4] build_version_e (model-free) ==="
python build_version_e.py --rules mined_transformations.py --parents public_molecules.smi --no-adversarial --max-parents 200 --per-parent-keep 10 --pop 30 --generations 15 --out version_e
echo "=== [3/4] assemble_benchmark ==="
python assemble_benchmark.py --authentic public_molecules.smi --min-subcat 4 --conformers --conf-cap 150 --out benchmark
echo "=== [4/4] generate_conformers ==="
python generate_conformers.py
echo "=== ALL DONE ==="
