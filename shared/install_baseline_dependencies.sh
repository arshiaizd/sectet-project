#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

python -m pip install \
  --requirement /mnt/vilab/scratch/arshia/projects/izadi/shared/baseline_requirements.txt

python - <<'PY'
import einops
import fitz
import seaborn
import yake

print("Baseline dependency verification passed")
print("einops:", einops.__version__)
print("PyMuPDF:", fitz.VersionBind)
print("seaborn:", seaborn.__version__)
print("yake:", yake.__version__ if hasattr(yake, "__version__") else "imported")
PY
