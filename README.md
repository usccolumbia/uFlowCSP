# uFlowCSP: Crystal Structure Prediction using Mean flow generative models

uFlowCSP is a deep learning-based generative model designed for crystal structure prediction (CSP) with a MeanFlow transport over an all-atom diffusion transformer. Introduced in September 2026 by researchers including Sourin Dey, Jianjun Hu and etc, it drastically accelerates the process of computational materials discovery. In general, it can 20-100X Faster Few-step crystal structure prediction with competitive or better accuracy. A full structure is decoded in 1–5 integration steps.

### University of South Carolina, Machine Learning and Evolution Lab.
### 09/23/2026, Developed by Sourin Dey and Jianjun Hu

Cite us: 
Dey, Sourin, Dipannoy Das Gupta, Lai Wei, Sadman Sadeed Omee, and Jianjun Hu. "uFlowCSP: Crystal Structure Prediction using Mean flow generative models." arXiv preprint arXiv:2609.09799 (2026). [Paper](https://arxiv.org/html/2609.09799v1)




Conditioning is **composition only** — the model is given a formula and generates
the lattice and every atomic position. No space group, no reference cell.

## Contents

1. [Install](#install)
2. [Data](#data) — one `git lfs pull`, everything else is derived
3. [Sample](#sample) — generate structures for a list of formulas
4. [Train](#train)
5. [End-to-end campaign](#end-to-end-campaign) — train → sample → evaluate, one command
6. [Troubleshooting](#troubleshooting)

---

## Install

```bash
conda create -n meanflow python=3.11 && conda activate meanflow
```

The Slurm scripts activate `meanflow` by name — use another name and pass
`CONDA_ENV=<name>`.

**1. PyTorch (CUDA 12.1 build).** Adjust `cu121` to match your driver.

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**2. PyG CUDA extensions.** Training only — skip for sampling.

```bash
pip install torch_scatter==2.1.2 torch_cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

**3. Everything else.**

```bash
pip install -r requirements.txt
```

**4. Verify.** Must print `2.5.1+cu121`. Plain `2.5.1` means step 3 replaced your
CUDA build — redo steps 1–2. ([why](#why-requirementstxt-omits-torch))

```bash
python -c "import torch; print(torch.__version__)"
```

**5. ORB relaxer.** Optional — only for `--relax`. Two commands, because
`orb-models` declares `torch>=2.6` while this stack is on 2.5.1:

```bash
pip install --no-deps orb-models==0.5.5
pip install cached_path dm-tree
```

---

## Data

**The MP-20 source file is stored in Git LFS.** A clone made without LFS leaves a
~130-byte pointer in its place, and every later step fails with an
unrelated-looking parse error. Run once, in the clone:

```bash
git lfs install && git lfs pull
```

`ls -l data/mp_20/raw/all.csv` must show ~135 MB, not ~130 bytes. That is the
only manual data step — everything else is built on demand.

#### What ships

| Path | What it is |
|---|---|
| `data/mp_20/raw/all.csv` | MP-20 source, one row per structure with `material_id`, `cif`, `spacegroup.number` (Git LFS, ~135 MB) |
| `data/splits/difCSP_{train,val,test}_ids.json` | The canonical CDVAE/DiffCSP split — 27136 / 9047 / 9046 material ids. Fixed; do not regenerate |
| `data/splits/difCSP_test.csv` | The 9046 test materials as `material_id, primitive_formula, spacegroup` |
| `180_primitive.csv` | The 180-formula CSPBench lane — a separate benchmark; never mix its numbers with the MP-20 ones |

`data/mp_20/train/*.arrow` and `dataset_dict.json` are a HuggingFace copy of the
same data. Nothing here reads them; `all.csv` is the only input.

#### What is derived

| Path | Built by | When |
|---|---|---|
| `data/mp20_raw/processed/mp20_raw_ordered_symmetry.pt` | the datamodule, automatically | first training run; slow (parses 45k cifs), then reused |
| `gt_cifs_test/` | `make_gt_cifs.py` | before any MP-20 evaluation |
| `data/splits/difCSP_test_knownz_full.csv` | `make_knownz_csv.py` | before any known-Z evaluation |

[The campaign](#end-to-end-campaign) builds the last two itself if they are
missing. To build them by hand:

```bash
python make_gt_cifs.py                       # -> gt_cifs_test/ (9046 cifs)
python make_knownz_csv.py \
    --csv data/splits/difCSP_test.csv \
    --gt_dir gt_cifs_test \
    --out data/splits/difCSP_test_knownz_full.csv
```

The known-Z CSV replaces each reduced formula with the true primitive-cell
contents (`Ti2O4`, not `TiO2`) — the DiffCSP / CrystalFlow convention, and what
makes the numbers comparable to theirs. Conditioning stays composition-only.

> The 180-formula lane also needs a `ground_truth_180/` folder of reference
> cells. It is **not** shipped and cannot be rebuilt from the files above (only
> 151 of its 180 materials exist in MP-20). The MP-20 campaign does not use it.

---

## Sample

#### Checkpoint

Not in this repository (~3.1 GB) — [train one](#train), or obtain a released one.
**Put it at `model.ckpt` in the repository root**; every command below assumes
that, and nothing in this repo refers to a path outside it.

Training writes `logs/<task>/runs/<run>/checkpoints/`. Take the
`uflow-best_valid-*.ckpt` with the highest `valid_rate`, not `last.ckpt` — the
selection is worth ~5 points. Renaming it to `model.ckpt` also avoids quoting the
`@` fields in its original name on every command line.

No config file is needed — architecture and hyperparameters live in the checkpoint.

#### Run

Formulas go in a CSV under a `primitive_formula` column, one per row:

```bash
printf 'primitive_formula\nSiO2\n' > test.csv

python src/match_meanflow_raw_all_formulas.py \
    --ckpt_path model.ckpt \
    --csv_path test.csv \
    --num_samples_per_formula 5 \
    --num_sampling_steps 5 \
    --save_generated_dir out/
```

Writes `out/SiO2/sample_0.cif` … `sample_4.cif`.

| Want to | Do |
|---|---|
| Sample more compositions | add rows to `test.csv`; `primitive_formula` is the only required column |
| Use your own CSV | `--csv_path <file>`, plus `--formula_col <name>` if the column is named differently. `180_primitive.csv` ships as an example |
| Relax the output (ORB v3, install step 5) | append `--relax` |
| Relax on CPU | `--relax --relax_device cpu --orb_model orb_v3_conservative_20_omat --relax_steps 50` |
| Condition on a space group too | `--use_spacegroup` (off by default — the reported runs are composition-only) |

ORB downloads its weights on first use, so that run needs internet. If compute
nodes are offline, warm the cache on the login node:

```bash
python -c "from orb_models.forcefield import pretrained; pretrained.orb_v3_conservative_20_omat(device='cpu')"
```

---

## Train

The shipped model — atom ordering + formula embedding + per-atom chemistry
features + coarse crystal-system token:

```bash
sbatch slurm/train_meanflow_raw_csp_ordered_formula_atomwise_system.sh
```

Those settings are the script's defaults; spelled out, they are
`0.25 1.5 difCSP false false false 0.0 symmetry false false true true true`. The
other three `train_meanflow_raw_csp_ordered*.sh` scripts are the ablation rungs
below it (ordering only, +formula, +atomwise), each a strict subset of the next.

Knobs, all environment variables:

| Variable | Default | Effect |
|---|---|---|
| `SEED` | 9 | random seed; the only thing varied across replicates |
| `MAX_EPOCHS` | 700 | training length |
| `BATCH_SIZE` | 256 | |
| `CKPT_PATH_OVERRIDE` | — | `null` forces a fresh run instead of resuming |
| `CAMPAIGN_LABEL` | — | pins the output directory name |
| `CONDA_ENV` | `meanflow` | env to activate |

Checkpoints land in `logs/<task>/runs/<run>/checkpoints/` as `last.ckpt` plus
`uflow-best_valid-…-valid_rate@<v>.ckpt`. **Sample from the best-validation file,
not `last.ckpt`** — half of all runs peak near epoch 499 and degrade by 699, and
the selection is worth ~5 points.

Logging is Weights & Biases, online by default: `wandb login` once, or export
`WANDB_API_KEY`, or set `WANDB_MODE=offline`, or append `logger=null`.

---

## End-to-end campaign

One command runs the whole lane — train → sample → evaluate, chained with Slurm
dependencies, one chain per seed:

```bash
bash slurm/submit_seed_replicates.sh        # repo root, on the login node
```

It pins the shipped configuration, resolves the best-validation checkpoint inside
each sampling job, samples the 9046 known-Z test materials at k=20 with S=1 and
S=5, and evaluates under the DiffCSP protocol (any-of-k, no relaxation, no energy
ranker, no space-group analysis, own candidates only). Missing data artefacts are
built on the way in. Results:

```
campaigns/seed_replicates/<CAMPAIGN>/eval/seed<N>_s<S>/summary.json
```

Read `diffcsp_match_rate` and `diffcsp_mean_rmse`. Check `n_formulas_total` is
9046 and `n_no_folder` is 0 — a non-zero `n_no_folder` means the sampling job was
truncated and the number is not comparable.

`SEEDS`, `K`, `STEPS_LIST`, `MAX_EPOCHS` and `PART` are environment overrides. The
script's header documents the reference numbers each row should land near, and
the ~2.6-point run-to-run spread that makes single-run comparisons meaningless.

---

## Troubleshooting

#### Why `requirements.txt` omits torch

It deliberately leaves out `torch`, `torchvision`, `torch_scatter` and
`torch_cluster`. Pinning them there makes pip re-resolve install steps 1–2, swap
the `+cu121` build for the default PyPI wheel, and fail building `torch_scatter`
from source with `ModuleNotFoundError: No module named 'torch'`.

#### `ModuleNotFoundError: No module named 'pkg_resources'`

setuptools is too new — `matminer` imports it and setuptools removed it in v81:

```bash
pip install "setuptools<81"
```

#### `ModuleNotFoundError` from inside `orb_models`

Install step 5's second command missed a dependency. List them all and install
everything except `torch`:

```bash
python -c "import importlib.metadata as m; print('\n'.join(m.requires('orb-models')))"
```

#### A data file looks empty or has the wrong columns

`data/mp_20/raw/all.csv` is a Git LFS pointer, not the real file. See [Data](#data).

#### Training dies immediately with a W&B error

The job has no Weights & Biases credentials. `wandb login`, or submit with
`WANDB_MODE=offline`.

#### A Slurm script fails with `sbatch: error: invalid partition`

The `#SBATCH -p` headers name this project's cluster partitions (`gpu-A100`,
`gpu-H200`). Override per submission with `sbatch -p <your-partition>`, or
`PART=<your-partition>` for the campaign.
