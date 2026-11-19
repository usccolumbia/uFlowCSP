## uFlowCSP: Crystal Structure Prediction using Mean Flow Generative Models

uFlowCSP predicts crystal structures from a chemical formula alone. It pairs a
MeanFlow transport with an all-atom diffusion transformer and decodes a full
structure (lattice and every atomic position) in 1–5 integration steps, rather
than the hundreds to thousands of steps many generative CSP models take.

Conditioning is **composition only**: no space group, no reference cell.

This branch holds the current model. Compared with the published version it
treats atomic positions as **periodic fractional coordinates** and gives the
network **lattice-plane features**:

- positions live on the torus [0, 1)³: uniform prior, shortest-way interpolation,
  and wrapping after every sampler step;
- the network sees each position as sin/cos(2πk·f) for k = 1…8 along each axis
  (48 numbers per atom) plus sin/cos(2π**G**·f) for the 56 mixed plane directions
  **G** = (h, k, l) with |h|, |k|, |l| ≤ 2;
- one lattice per crystal (the Niggli cell, 9 numbers), with the loss weighted
  10 : 1 between positions and lattice.

The published model (Cartesian positions, arXiv:2609.09799) is on the
[`cartesian`](https://github.com/usccolumbia/uFlowCSP/tree/cartesian) branch,
together with its figshare checkpoint.

Developed by Sourin Dey and Jianjun Hu, Machine Learning and Evolution Lab,
University of South Carolina.

## Results

Test sets, 5 sampling steps, no relaxation, known-Z formulas. Mean ± standard
deviation over three training seeds; the denominator is every test material.
Match rates and RMSE follow DiffCSP's evaluation code (also used by CrystalFlow),
including its validity check; a match is `StructureMatcher(stol=0.5, ltol=0.3,
angle_tol=10)`.

| Test set | Model | k = 1 match rate (%) | k = 20 match rate (%) | RMSE (k = 1) | RMSE (k = 20) |
|---|---|---|---|---|---|
| MP-20 (9,046) | 700 epochs (final checkpoint) | 61.02 ± 0.38 | 77.74 ± 0.03 | 0.0645 ± 0.0020 | 0.0467 ± 0.0016 |
| MPTS-52 (8,096) | 1,200 epochs (chosen on validation) | 27.04 ± 0.29 | 36.22 ± 0.32 | 0.1130 ± 0.0044 | 0.1085 ± 0.0024 |

`scripts/benchmark_mp20.sh` and `scripts/benchmark_mpts52.sh` score a checkpoint
on these test sets (`filtered_*` keys in the summary).

## Contents

**Part 1 — Generate structures** (a GPU and ~10 minutes of setup)

1. [Install](#1-install)
2. [Pretrained models](#2-pretrained-models)
3. [Sample](#3-sample)
4. [Options](#4-options)

**Part 2 — Train and benchmark** (one GPU)

5. [Extra setup](#5-extra-setup)
6. [Train](#6-train)
7. [Benchmark](#7-benchmark)

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

## 2. Pretrained models

Two checkpoints, one per benchmark (3.2 GB each):

| Checkpoint | Trained on | Use for |
|---|---|---|
| `model_mp20.ckpt` | MP-20 (≤ 20 atoms per cell), 700 epochs | cells up to ~20 atoms |
| `model_mpts52.ckpt` | MPTS-52 (≤ 52 atoms per cell), 1,200 epochs | larger cells |

**Download links will be added here soon.** Until then, train your own model
([Part 2](#part-2--train-and-benchmark)).

No config file is needed: the architecture and hyperparameters are stored in
the checkpoint. For the published (Cartesian) checkpoint, use the `cartesian`
branch.

## 3. Sample

List formulas in a CSV with a `primitive_formula` column, then run the sampler:

```bash
printf 'primitive_formula\nSiO2\nNaCl\nSrTiO3\n' > test.csv

python src/uflowcsp.py \
    --ckpt_path model_mp20.ckpt \
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

Samples that fail the validity check (any two atoms closer than 0.5 Å, or a cell
volume below 0.1 Å³) are dropped, so a folder can hold fewer than
`--num_samples_per_formula` files.

**What the formula means.** The formula sets the exact atom count of the
generated cell: `SiO2` gives a 3-atom cell, `Si3O6` a 9-atom cell. Write out the
full cell contents if you want a larger cell. Use the MP-20 model for cells up to
~20 atoms and the MPTS-52 model for larger ones.

Generated structures are **not relaxed**. Relax them before computing
properties — see `--relax` below.

## 4. Options

| Want to | Do |
|---|---|
| Skip the CSV | `--formula SiO2` (or several: `--formula SiO2 NaCl`) instead of `--csv_path` |
| Sample more formulas | add rows to the CSV |
| Use a CSV with a different column name | `--formula_col <name>` (`180_primitive.csv` ships as an example) |
| Sample faster | `--num_sampling_steps 1` (one-step decoding) |
| Relax the output with ORB v3 | install ORB (below), then append `--relax` |
| Relax on CPU | `--relax --relax_device cpu --orb_model orb_v3_conservative_20_omat --relax_steps 50` |
| See every option | `python src/uflowcsp.py --help` |

**Installing the ORB relaxer.** Two commands, because `orb-models` declares
`torch>=2.6` while this stack is on 2.5.1:

```bash
pip install --no-deps orb-models==0.5.5
pip install cached_path dm-tree
```

ORB downloads its weights on first use, so that first run needs internet. To
download them ahead of time:

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
`WANDB_API_KEY`, or set `WANDB_MODE=offline` to log locally. Do not switch the
logger off (`logger=null`): no checkpoints are written without it.

<details>
<summary>Data files: what ships and what is derived</summary>

| Path | What it is |
|---|---|
| `data/mp_20/raw/all.csv` | MP-20 source, one row per structure with `material_id`, `cif`, `spacegroup.number` (Git LFS, ~135 MB) |
| `data/splits/difCSP_{train,val,test}_ids.json` | The canonical CDVAE/DiffCSP split — 27136 / 9047 / 9046 material ids. Fixed; do not regenerate |
| `data/splits/difCSP_test.csv` | The 9046 test materials as `material_id, primitive_formula, spacegroup` |
| `180_primitive.csv` | The 180-formula CSPBench lane — a separate benchmark from MP-20 |

`data/mp_20/train/*.arrow` and `dataset_dict.json` are a HuggingFace copy of the
same data. Nothing here reads them; `all.csv` is the only input.

| Derived path | Built by | When |
|---|---|---|
| `data/mp20_raw/processed/mp20_raw_ordered_symmetry.pt` | the datamodule, automatically | first training run; slow (parses 45k cifs), then reused |
| `gt_cifs_test/` | `make_gt_cifs.py` | before any MP-20 evaluation |
| `data/splits/difCSP_test_knownz_full.csv` | `make_knownz_csv.py` | before any known-Z evaluation |

`scripts/benchmark_mp20.sh` builds the last two itself if they are missing. To build them by
hand:

```bash
python make_gt_cifs.py                       # -> gt_cifs_test/ (9046 cifs)
python make_knownz_csv.py \
    --csv data/splits/difCSP_test.csv \
    --gt_dir gt_cifs_test \
    --out data/splits/difCSP_test_knownz_full.csv
```

The known-Z CSV replaces each reduced formula with the true primitive-cell
contents (`Ti2O4`, not `TiO2`) — the DiffCSP / CrystalFlow convention.
Conditioning stays composition-only.

The 180-formula lane also needs a `ground_truth_180/` folder of reference cells.
It is **not** shipped and cannot be rebuilt from the files above (only 151 of its
180 materials exist in MP-20). The MP-20 benchmark does not use it.

</details>

**The MPTS-52 data** is not shipped. Build it once from DiffCSP's split CSVs:

```bash
git clone --depth 1 https://github.com/jiaor17/DiffCSP.git /tmp/DiffCSP
python scripts/prepare_mpts52.py --diffcsp_dir /tmp/DiffCSP/data/mpts_52
```

This writes `data/mpts/raw/all.csv`, `data/splits/mpts_{train,val,test}_ids.json`,
the reference CIFs `gt_cifs_mpts52_test/` and the test-target list
`data/splits/mpts52_test_knownz.csv`.

## 6. Train

From the repository root:

```bash
bash scripts/train.sh            # MP-20, 700 epochs
bash scripts/train_mpts52.sh     # MPTS-52, 1,200 epochs (after prepare_mpts52.py)
```

The two runs differ only in the data and in the MeanFlow time-pair law
(MP-20: logistic-normal; MPTS-52: uniform on [0, 1]). The 180 CSPBench formulas
(`180_primitive.csv`) are excluded from MP-20 training. Append any Hydra override
at the end. To train without a Weights & Biases account, set `WANDB_MODE=offline`.
Other settings are environment variables:

| Variable | Default | Effect |
|---|---|---|
| `SEED` | 9 | random seed |
| `MAX_EPOCHS` | 700 (MP-20), 1200 (MPTS-52) | training length |
| `BATCH_SIZE` × `ACCUM` | 128 × 2 | effective batch 256; fits a 24 GB GPU. With 40 GB or more, `BATCH_SIZE=256 ACCUM=1` is equivalent |
| `RESUME` | — | path to a `last.ckpt` to continue a run |
| `PLANE_FEATURES` | true | `false` trains with the 48 per-axis features only (ablation) |

**Checkpoints.** Training writes `logs/train_uflowcsp_<dataset>/runs/<run>/checkpoints/`;
every 100th epoch is kept in `periodic/`. The reported models are
`periodic/uflow-periodic-epoch@699.ckpt` (MP-20, the final checkpoint) and
`periodic/uflow-periodic-epoch@1199.ckpt` (MPTS-52). Copy the one you want to
`model_mp20.ckpt` or `model_mpts52.ckpt`; the rename also avoids quoting the `@`.

## 7. Benchmark

Score a checkpoint on a test set under the DiffCSP / CrystalFlow protocol (k
candidates per material, no relaxation, each material scored against only its own
candidates):

```bash
bash scripts/benchmark_mp20.sh model_mp20.ckpt         # 9,046 MP-20 test materials
bash scripts/benchmark_mpts52.sh model_mpts52.ckpt     # 8,096 MPTS-52 test materials
```

The defaults are k = 20 candidates and S = 5 steps; set `K` and `STEPS` to change
them. Results go to `results/<dataset>_S<S>_k<K>/eval/summary.json`:

| Key | Meaning |
|---|---|
| `filtered_k1_match_rate`, `filtered_k20_match_rate` | candidates must pass DiffCSP's validity check (as in its `compute_metrics.py`); at k = 1 the reference must be valid too |
| `filtered_k1_rmse`, `filtered_k20_rmse` | mean RMSE over matched materials |
| `unfiltered_*` | the same, with structural validity only |

A match is `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3).get_rms_dist(...)`
not returning `None`. `evaluate_diffcsp.py` can also score any folder of
`<material_id>_sample_<i>.cif` files.

Results vary between training seeds (MP-20 k = 1: about ±0.4 points over three
seeds). To estimate the spread, train with several values of `SEED` and benchmark
each run.

---

## Troubleshooting

#### Warnings printed during sampling

A `pkg_resources is deprecated` warning and a pymatgen `gcd is deprecated`
warning are expected.

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

#### A data file looks empty or has the wrong columns

`data/mp_20/raw/all.csv` is a Git LFS pointer, not the real file. See
[Extra setup](#5-extra-setup).

#### Training dies immediately with a W&B error

No Weights & Biases credentials are set. Run `wandb login`, or train with
`WANDB_MODE=offline`.

---

## Citation

A paper describing this version is in preparation. Until then, please cite the
uFlowCSP paper:

> Dey, Sourin, Dipannoy Das Gupta, Lai Wei, Sadman Sadeed Omee, and Jianjun Hu.
> "uFlowCSP: Crystal Structure Prediction using Mean flow generative models."
> arXiv preprint arXiv:2609.09799 (2026).
> [Paper](https://arxiv.org/html/2609.09799v1)

Code is released under the MIT License.
