## uFlowCSP: Crystal Structure Prediction using Mean Flow Generative Models

uFlowCSP predicts crystal structures from a chemical formula alone. It pairs a
MeanFlow transport with an all-atom diffusion transformer and decodes a full
structure — lattice and every atomic position — in 1–5 integration steps, 20–100×
faster than many-step generative CSP models at competitive or better accuracy.

Conditioning is **composition only**: no space group, no reference cell.

Developed by Sourin Dey and Jianjun Hu, Machine Learning and Evolution Lab,
University of South Carolina (September 2026).
[Paper (arXiv:2609.09799)](https://arxiv.org/html/2609.09799v1) ·
[Pretrained model (figshare)](https://doi.org/10.6084/m9.figshare.33974422)

## Contents

**Part 1 — Generate structures** (a GPU and ~10 minutes of setup)

1. [Install](#1-install)
2. [Download the pretrained model](#2-download-the-pretrained-model)
3. [Sample](#3-sample)
4. [Options](#4-options)

**Part 2 — Train and benchmark** (Slurm cluster)

5. [Extra setup](#5-extra-setup)
6. [Train](#6-train)
7. [End-to-end benchmark campaign](#7-end-to-end-benchmark-campaign)

[Troubleshooting](#troubleshooting) · [Citation](#citation)

---

# Part 1 — Generate structures

## 1. Install

```bash
git clone https://github.com/usccolumbia/uFlowCSP.git
cd uFlowCSP

conda create -n meanflow python=3.11 -y
conda activate meanflow

# PyTorch, CUDA 12.1 build (change cu121 to match your driver)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# everything else
pip install -r requirements.txt
```

Check the install. This must print `2.5.1+cu121`:

```bash
python -c "import torch; print(torch.__version__)"
```

If it prints plain `2.5.1`, the requirements step replaced your CUDA build —
rerun the PyTorch line. ([why](#why-requirementstxt-omits-torch))

## 2. Download the pretrained model

The checkpoint (3.1 GB, trained on MP-20) is on figshare:
[doi:10.6084/m9.figshare.33974422](https://doi.org/10.6084/m9.figshare.33974422).
Save it as `model.ckpt` in the repository root:

```bash
curl -L -o model.ckpt https://ndownloader.figshare.com/files/69274087
md5sum model.ckpt    # expect c1ff449289976c793b06bdbfcc9659ad
```

No config file is needed — the architecture and hyperparameters are stored in
the checkpoint.

## 3. Sample

List formulas in a CSV with a `primitive_formula` column, then run the sampler:

```bash
printf 'primitive_formula\nSiO2\nNaCl\nSrTiO3\n' > test.csv

python src/match_meanflow_raw_all_formulas.py \
    --ckpt_path model.ckpt \
    --csv_path test.csv \
    --num_samples_per_formula 5 \
    --num_sampling_steps 5 \
    --save_generated_dir out/
```

Output:

```
out/
├── generation_summary.csv      # per formula: generated / valid / saved counts
├── NaCl/sample_0.cif … sample_4.cif
├── SiO2/sample_0.cif … sample_4.cif
└── SrTiO3/sample_0.cif … sample_4.cif
```

This takes under a minute on one RTX 3090, most of it loading the model. Samples
that fail the validity check are dropped, so a folder can hold fewer than
`--num_samples_per_formula` files.

**What the formula means.** The formula sets the exact atom count of the
generated cell: `SiO2` gives a 3-atom cell, `Si3O6` a 9-atom cell. Write out the
full cell contents if you want a larger cell. The model was trained on MP-20
(at most 20 atoms per cell) and is most reliable on small cells; expect lower
quality above ~20 atoms.

Generated structures are **not relaxed**. Relax them before computing
properties — see `--relax` below.

## 4. Options

| Want to | Do |
|---|---|
| Sample more formulas | add rows to the CSV |
| Use a CSV with a different column name | `--formula_col <name>` (`180_primitive.csv` ships as an example) |
| Sample faster | `--num_sampling_steps 1` (one-step decoding) |
| Relax the output with ORB v3 | install ORB (below), then append `--relax` |
| Relax on CPU | `--relax --relax_device cpu --orb_model orb_v3_conservative_20_omat --relax_steps 50` |
| Also condition on a space group | `--use_spacegroup` (off by default; reported results are composition-only) |
| See every option | `python src/match_meanflow_raw_all_formulas.py --help` |

**Installing the ORB relaxer.** Two commands, because `orb-models` declares
`torch>=2.6` while this stack is on 2.5.1:

```bash
pip install --no-deps orb-models==0.5.5
pip install cached_path dm-tree
```

ORB downloads its weights on first use, so that first run needs internet. If
your compute nodes are offline, warm the cache on the login node:

```bash
python -c "from orb_models.forcefield import pretrained; pretrained.orb_v3_conservative_20_omat(device='cpu')"
```

---

# Part 2 — Train and benchmark

## 5. Extra setup

Everything in Part 1, plus:

**PyG CUDA extensions** (needed for training):

```bash
pip install torch_scatter==2.1.2 torch_cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

**The MP-20 data** is stored in Git LFS. Without LFS, the clone holds a
~130-byte pointer instead of the file, and later steps fail with an
unrelated-looking parse error. Run once, in the clone:

```bash
git lfs install && git lfs pull
ls -l data/mp_20/raw/all.csv    # must be ~135 MB, not ~130 bytes
```

That is the only manual data step — everything else is built on demand.

**Weights & Biases** logging is on by default: `wandb login` once, or export
`WANDB_API_KEY`, or set `WANDB_MODE=offline`, or append `logger=null`.

**Slurm partitions.** The `#SBATCH -p` headers name this project's partitions
(`gpu-A100`, `gpu-H200`). Override with `sbatch -p <your-partition>`, or
`PART=<your-partition>` for the campaign. The scripts activate the `meanflow`
env by name; pass `CONDA_ENV=<name>` if yours differs.

<details>
<summary>Data files: what ships and what is derived</summary>

| Path | What it is |
|---|---|
| `data/mp_20/raw/all.csv` | MP-20 source, one row per structure with `material_id`, `cif`, `spacegroup.number` (Git LFS, ~135 MB) |
| `data/splits/difCSP_{train,val,test}_ids.json` | The canonical CDVAE/DiffCSP split — 27136 / 9047 / 9046 material ids. Fixed; do not regenerate |
| `data/splits/difCSP_test.csv` | The 9046 test materials as `material_id, primitive_formula, spacegroup` |
| `180_primitive.csv` | The 180-formula CSPBench lane — a separate benchmark; never mix its numbers with the MP-20 ones |

`data/mp_20/train/*.arrow` and `dataset_dict.json` are a HuggingFace copy of the
same data. Nothing here reads them; `all.csv` is the only input.

| Derived path | Built by | When |
|---|---|---|
| `data/mp20_raw/processed/mp20_raw_ordered_symmetry.pt` | the datamodule, automatically | first training run; slow (parses 45k cifs), then reused |
| `gt_cifs_test/` | `make_gt_cifs.py` | before any MP-20 evaluation |
| `data/splits/difCSP_test_knownz_full.csv` | `make_knownz_csv.py` | before any known-Z evaluation |

The campaign builds the last two itself if they are missing. To build them by
hand:

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

The 180-formula lane also needs a `ground_truth_180/` folder of reference cells.
It is **not** shipped and cannot be rebuilt from the files above (only 151 of its
180 materials exist in MP-20). The MP-20 campaign does not use it.

</details>

## 6. Train

The released model — atom ordering + formula embedding + per-atom chemistry
features + coarse crystal-system token:

```bash
sbatch slurm/train_meanflow_raw_csp_ordered_formula_atomwise_system.sh
```

Its settings are the script's defaults; spelled out, they are
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

**Using your own checkpoint.** Training writes
`logs/<task>/runs/<run>/checkpoints/` with `last.ckpt` plus
`uflow-best_valid-…-valid_rate@<v>.ckpt` files. Sample from the best-validation
file with the highest `valid_rate`, **not** `last.ckpt` — half of all runs peak
near epoch 499 and degrade by 699, and the selection is worth ~5 points. Copy it
to `model.ckpt` and follow [Sample](#3-sample); the rename also avoids quoting
the `@` in its name.

## 7. End-to-end benchmark campaign

One command runs train → sample → evaluate, chained with Slurm dependencies,
one chain per seed:

```bash
bash slurm/submit_seed_replicates.sh        # repo root, on the login node
```

It pins the released configuration, picks the best-validation checkpoint inside
each sampling job, samples the 9046 known-Z test materials at k=20 with S=1 and
S=5 steps, and evaluates under the DiffCSP protocol (any-of-k, no relaxation, no
energy ranker, no space-group analysis, own candidates only). Missing data files
are built on the way in. Results:

```
campaigns/seed_replicates/<CAMPAIGN>/eval/seed<N>_s<S>/summary.json
```

Read `diffcsp_match_rate` and `diffcsp_mean_rmse`. Check that `n_formulas_total`
is 9046 and `n_no_folder` is 0 — a non-zero `n_no_folder` means the sampling job
was truncated and the number is not comparable.

`SEEDS`, `K`, `STEPS_LIST`, `MAX_EPOCHS` and `PART` are environment overrides.
The script's header documents the reference numbers each row should land near,
and the ~2.6-point run-to-run spread that makes single-run comparisons
meaningless.

---

## Troubleshooting

#### Warnings printed during sampling

A `pkg_resources is deprecated` warning, a pymatgen `gcd is deprecated` warning,
and a long note about `meanflow.lattice_per_structure=False` are all expected.
The last one explains a setting the released checkpoint was trained with; no
action is needed to sample from it.

#### Why `requirements.txt` omits torch

It deliberately leaves out `torch`, `torchvision`, `torch_scatter` and
`torch_cluster`. Pinning them there makes pip re-resolve the PyTorch install,
swap the `+cu121` build for the default PyPI wheel, and fail building
`torch_scatter` from source with `ModuleNotFoundError: No module named 'torch'`.

#### `ModuleNotFoundError: No module named 'pkg_resources'`

setuptools is too new — `matminer` imports it and setuptools removed it in v81:

```bash
pip install "setuptools<81"
```

#### `ModuleNotFoundError` from inside `orb_models`

The ORB install missed a dependency. List them all and install everything except
`torch`:

```bash
python -c "import importlib.metadata as m; print('\n'.join(m.requires('orb-models')))"
```

#### `model.ckpt` fails to load

The download was probably interrupted. Check `md5sum model.ckpt` against
`c1ff449289976c793b06bdbfcc9659ad` and download again if it differs.

#### A data file looks empty or has the wrong columns

`data/mp_20/raw/all.csv` is a Git LFS pointer, not the real file. See
[Extra setup](#5-extra-setup).

#### Training dies immediately with a W&B error

The job has no Weights & Biases credentials. `wandb login`, or submit with
`WANDB_MODE=offline`.

#### A Slurm script fails with `sbatch: error: invalid partition`

See **Slurm partitions** under [Extra setup](#5-extra-setup).

---

## Citation

If you use uFlowCSP, please cite the paper:

> Dey, Sourin, Dipannoy Das Gupta, Lai Wei, Sadman Sadeed Omee, and Jianjun Hu.
> "uFlowCSP: Crystal Structure Prediction using Mean flow generative models."
> arXiv preprint arXiv:2609.09799 (2026).
> [Paper](https://arxiv.org/html/2609.09799v1)

and, if you use the released checkpoint:

> Dey, Sourin (2026). uFlowCSP: Fast Crystal Structure Prediction using Mean flow
> generative models. figshare.
> https://doi.org/10.6084/m9.figshare.33974422.v1

Code is released under the MIT License; the pretrained model under CC BY 4.0.
