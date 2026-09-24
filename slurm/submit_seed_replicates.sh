#!/bin/bash
# =============================================================================
# submit_seed_replicates.sh
#   THE END-TO-END CAMPAIGN. Seed replicates of the SHIPPED _system config, each
#   a fully chained pipeline with NO manual per-stage submission:
#
#        train (fresh, seed=S)  --afterok-->  sample (k,S)  --afterok-->  eval
#
#   Training is always
#       slurm/train_meanflow_raw_csp_ordered_formula_atomwise_system.sh
#   (hardcoded, not overridable: the reported numbers below belong to that model
#   and nothing else). Its arguments are pinned explicitly here rather than left
#   to the script's defaults, so a later change to those defaults cannot silently
#   retune this campaign.
#
#   THE SETUP IS THE BEST ONE ESTABLISHED ON mean_flow_v2_csp — the Table 1 lane:
#     * model       : MeanFlow ordered (symmetry) + formula embedding + atomwise
#                     chemistry features + coarse crystal-system token,
#                     fine 230-way SG NOT used
#     * training    : cfg_dropout=0.8, cfg_scale=1.5, flow_ratio=0.25, difCSP
#                     split, lr=1e-4, batch 256, 700 epochs, shipped lognorm
#                     (-0.4, 1.0) time distribution, fresh (no resume)
#     * checkpoint  : CKPT_MODE=bestvalid — the highest-valid_rate
#                     uflow-best_valid-*.ckpt, resolved INSIDE the sampling job.
#                     Worth +5.2 pts at S=5 over last.ckpt on MP-20: half the
#                     runs peak at ep499 and degrade by ep699.
#     * sampling    : known-Z difCSP test CSV (n=9046), k=20, S in {1, 5},
#                     RAW (no relaxation), formula-only (GT space group never
#                     fed), material-prefixed filenames
#     * eval        : DiffCSP CSP-task protocol — ltol=0.3 stol=0.5 angle_tol=10,
#                     any-of-k, per material, NO energy ranker, NO spacegroup
#                     analysis, and OWN CANDIDATES ONLY
#
#   Alternatives that were TRIED ON v2_csp AND LOST, so they are not options
#   here: uniform (t, r) time distribution (80.48 vs 81.68 = null), cfg_dropout
#   0.1/0.5 (flat), noise-temperature sweep (flat), crystal-system enumeration
#   at sampling time (flat), ORB relaxation before matching (+0.9 pt, and not
#   what the papers do). The shipped configuration above is the one to submit.
#
#   WHY REPLICATES: two runs of the identical config (seed 9, 700 ep) gave
#   known-Z k=20 S=5 match rates of 72.81% and 82.69% — a 9.9-pt gap with no
#   config difference. Under the FIXED bestvalid rule the residual seed spread
#   is ~2.6 pts, but that still swamps the +2-pt increments in the ablation
#   ladder. Never headline a single run on this metric.
#
#   ANCHORS — diffcsp known-Z lane, k=20, n=9046, own candidates, bestvalid:
#       ours  S=5 (100 NFE)   81.68 +/- 2.63 / RMSE 0.0673 +/- 0.0067  (n=4)
#       ours  S=1 ( 20 NFE)   76.30 +/- 2.62 / RMSE 0.0826 +/- 0.0042  (n=4)
#       CrystalFlow           78.34 / RMSE 0.0577   @  2,000 NFE
#       DiffCSP               77.93 / RMSE 0.0492   @ ~20,000 NFE
#   (79.3 +/- 4.1 is the OLD last.ckpt aggregate — do NOT compare a bestvalid
#   run against it.)
#
#   Run on the CLUSTER LOGIN NODE, from the repo root:
#       bash slurm/submit_seed_replicates.sh
#
#   Results land in  campaigns/seed_replicates/<CAMPAIGN>/eval/seed<S>_s<STEPS>/summary.json
#   Collect with the greps printed at the end.
# =============================================================================
set -euo pipefail

# ---- SWEEP + KNOBS (edit here) ----------------------------------------------
SEEDS=(${SEEDS:-10 11 12 13})    # config's own seed is 9 -> those runs already exist
K=${K:-20}                       # samples/material = any-of-k (headline k=20)
# Each seed is sampled+evaluated at EVERY value here, all branching off the same
# (expensive) training job: S=5 -> 100 NFE/material, S=1 -> 20 NFE/material (the
# true 1-step row). Both go in Table 1.
STEPS_LIST=(${STEPS_LIST:-1 5})
BATCH=${BATCH:-100}              # generation batch size
MAX_EPOCHS=${MAX_EPOCHS:-700}    # must match the existing runs for a valid replicate
CKPT_MODE=${CKPT_MODE:-bestvalid}   # bestvalid | last  -- bestvalid is the Table 1 rule
PART=${PART:-gpu-A100}
TRAIN_TIME=${TRAIN_TIME:-}       # optional --time for training (e.g. 24:00:00)

# ---- the shipped training configuration, pinned ------------------------------
# Positional args of the _system train script, in order:
#   flow_ratio cfg_scale split_type sym_lattice sym_rotavg sym_anchor
#   sym_loss_weight atom_ordering perm_augment modulo_translation
#   use_formula_embedding use_atomwise_features use_crystal_system
TRAIN_ARGS=(0.25 1.5 difCSP false false false 0.0 symmetry false false true true true)
CFG_DROPOUT=${CFG_DROPOUT:-0.8}  # drives BOTH cfg_ratio and class_dropout_prob
TRAIN_BATCH=${TRAIN_BATCH:-256}

# ---- fixed paths ------------------------------------------------------------
# NOT overridable: every number quoted above belongs to this training script.
TRAIN_SCRIPT="slurm/train_meanflow_raw_csp_ordered_formula_atomwise_system.sh"
GEN_PY="src/match_meanflow_raw_all_formulas.py"
EVAL_SCRIPT="slurm/select_top5_and_evaluate.sh"
TRAIN_RUNS_DIR="logs/train_meanflow_raw_csp_ordered_formula_atomwise_system/runs"
FULL_CSV=${FULL_CSV:-data/splits/difCSP_test.csv}
KNOWNZ_CSV=${KNOWNZ_CSV:-data/splits/difCSP_test_knownz_full.csv}
GT_DIR=${GT_DIR:-gt_cifs_test}

CAMPAIGN=${CAMPAIGN:-seedrep_$(date +%d%b_%H%M)}
ROOT="campaigns/seed_replicates/${CAMPAIGN}"
GEN_ROOT="${ROOT}/gen"; EVAL_ROOT="${ROOT}/eval"; LOG_DIR="${ROOT}/logs"
WRAP_DIR="${ROOT}/wrappers"
REG="${ROOT}/ckpt_registry.sh"; MANIFEST="${ROOT}/MANIFEST.txt"

# ---- preflight --------------------------------------------------------------
for f in "$TRAIN_SCRIPT" "$GEN_PY" "$EVAL_SCRIPT"; do
    [ -f "$f" ] || { echo "ABORT: missing $f (run from the repo root)"; exit 1; }
done
grep -q 'seed=${SEED:-9}' "$TRAIN_SCRIPT" || {
    echo "ABORT: $TRAIN_SCRIPT has no SEED support — every run would use seed 9."; exit 1; }
case "$CKPT_MODE" in bestvalid|last) : ;;
    *) echo "ABORT: CKPT_MODE must be bestvalid or last (got '$CKPT_MODE')"; exit 1 ;;
esac
[ "$CKPT_MODE" = "bestvalid" ] || echo "NOTE: CKPT_MODE=last — on MP-20 that costs ~5.2 pts at S=5."

ALLATOM_PY="${ALLATOM_PY:-${PYTHON:-python}}"

# data/mp_20/raw/all.csv is a Git LFS object. A clone made without LFS leaves a
# ~130-byte pointer, which otherwise surfaces as an unrelated parse error deep
# inside the first training job.
RAW_CSV="data/mp_20/raw/all.csv"
if [ -f "$RAW_CSV" ] && [ "$(wc -c < "$RAW_CSV")" -lt 1024 ] \
   && head -c 40 "$RAW_CSV" | grep -q 'git-lfs'; then
    echo "ABORT: $RAW_CSV is a Git LFS pointer, not the real file."
    echo "       Run: git lfs install && git lfs pull"
    exit 1
fi

# Build the ground-truth cifs once (one <material_id>.cif per test material).
# Needed BOTH by the known-Z CSV builder (it reads the true cell contents out of
# these) and by the evaluator (it matches against them).
if [ ! -d "$GT_DIR" ] || [ -z "$(ls -A "$GT_DIR" 2>/dev/null)" ]; then
    echo "== building GT cifs -> ${GT_DIR}/  (a few minutes; reads ${RAW_CSV}) =="
    PYTHONNOUSERSITE=1 "$ALLATOM_PY" make_gt_cifs.py \
        --test_ids data/splits/difCSP_test_ids.json --out_dir "$GT_DIR" \
        || { echo "ABORT: GT cif build failed"; exit 1; }
fi
GT_COUNT=$(ls -1 "$GT_DIR"/*.cif 2>/dev/null | wc -l)
[ "$GT_COUNT" -gt 0 ] || { echo "ABORT: no cifs in $GT_DIR (known-Z needs the GT cells)"; exit 1; }
echo "GT cifs: ${GT_COUNT} in ${GT_DIR}/"

# Build the known-Z CSV once (full cell contents read from the GT cifs).
if [ ! -f "$KNOWNZ_CSV" ]; then
    [ -f "$FULL_CSV" ] || { echo "ABORT: base csv not found: $FULL_CSV"; exit 1; }
    echo "== building known-Z CSV -> ${KNOWNZ_CSV} =="
    PYTHONNOUSERSITE=1 "$ALLATOM_PY" make_knownz_csv.py \
        --csv "$FULL_CSV" --gt_dir "$GT_DIR" --out "$KNOWNZ_CSV" \
        || { echo "ABORT: known-Z CSV build failed"; exit 1; }
    if head -5 "$KNOWNZ_CSV" | grep -q '[A-Za-z0-9] [A-Z]'; then
        echo "ABORT: known-Z CSV has SPACED formulas — fix make_knownz_csv.py"; rm -f "$KNOWNZ_CSV"; exit 1
    fi
else
    echo "known-Z CSV exists: $KNOWNZ_CSV (reusing)"
fi

# Guard the denominator. A subset CSV at this path (e.g. an old subset500 pilot)
# would silently shrink n_formulas_total and inflate every match rate — the exact
# failure mode that had to be ruled out by hand on the cd0p8 result.
EXPECT_ROWS=${EXPECT_ROWS:-9046}
KNOWNZ_ROWS=$(( $(grep -c '' "$KNOWNZ_CSV") - 1 ))   # data rows, header excluded
if [ "$KNOWNZ_ROWS" -ne "$EXPECT_ROWS" ]; then
    echo "ABORT: ${KNOWNZ_CSV} has ${KNOWNZ_ROWS} data rows, expected ${EXPECT_ROWS}."
    echo "       Full difCSP known-Z test set = 9046 materials. A subset here would"
    echo "       inflate the reported match rate. Rebuild it, or pass EXPECT_ROWS=<n>"
    echo "       if you deliberately want a subset run."
    exit 1
fi
echo "known-Z CSV verified: ${KNOWNZ_ROWS} materials (denominator will be ${KNOWNZ_ROWS})"

mkdir -p "$GEN_ROOT" "$EVAL_ROOT" "$LOG_DIR" "$WRAP_DIR"; : > "$MANIFEST"

# =============================================================================
# SAMPLING WRAPPER — written into the campaign dir, so no repo file is touched.
#
# It exists because the best-validation checkpoint's filename carries its own
# valid_rate (uflow-best_valid-epoch@599-...-valid_rate@0.8730.ckpt), so the path
# does not exist yet when this dependent job is submitted. The wrapper resolves
# it at RUN time, after training has finished.
# =============================================================================
GEN_WRAP="${WRAP_DIR}/gen.sh"
cat > "$GEN_WRAP" <<'SH'
#!/bin/bash
# gen.sh <CKDIR> <CKPT_MODE> <CSV> <K> <BATCH> <STEPS> <OUT_DIR>
module load python3/anaconda/3.12 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
[ -f ~/.bashrc ] && . ~/.bashrc
conda activate "${CONDA_ENV:-meanflow}" 2>/dev/null || true
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
ulimit -n 8192
PY="${ALLATOM_PY:-${PYTHON:-python}}"

CKDIR="$1"; MODE="$2"; CSV="$3"; K="$4"; BATCH="$5"; STEPS="$6"; OUT="$7"

if [ "$MODE" = "bestvalid" ]; then
    # Highest valid_rate wins; the rate is the last @-field of the filename.
    CKPT=$(ls -1 "$CKDIR"/uflow-best_valid-*.ckpt 2>/dev/null \
           | awk -F 'valid_rate@' '{print $2"\t"$0}' | sort -k1,1 -gr | head -1 | cut -f2- || true)
    if [ -z "$CKPT" ]; then
        echo "WARN: no uflow-best_valid-*.ckpt in $CKDIR -- falling back to last.ckpt." >&2
        echo "WARN: this row is NOT best-val selected; on MP-20 that fallback cost -5.2 pts." >&2
        CKPT="$CKDIR/last.ckpt"
    fi
else
    CKPT="$CKDIR/last.ckpt"
fi
[ -f "$CKPT" ] || { echo "ABORT: checkpoint not found: $CKPT" >&2; exit 1; }

echo "checkpoint mode: $MODE"
echo "checkpoint:      $CKPT"
echo "csv:             $CSV"
echo "out:             $OUT"
ls -1 "$CKDIR"/uflow-best_valid-*.ckpt 2>/dev/null || true

# Known-Z headline protocol: RAW structures (the papers match unrelaxed),
# material-prefixed filenames so each material owns its candidates, and NO
# space-group conditioning (formula-only inference).
PYTHONNOUSERSITE=1 "$PY" src/match_meanflow_raw_all_formulas.py \
    --ckpt_path "$CKPT" --csv_path "$CSV" \
    --num_samples_per_formula "$K" --batch_size "$BATCH" \
    --num_sampling_steps "$STEPS" --noise_scale 1.0 \
    --save_generated_dir "$OUT" \
    --disable_spacegroup --material_prefix
echo "gen DONE: $OUT"
SH
chmod +x "$GEN_WRAP"

echo "============================================================"
echo " END-TO-END SEED REPLICATES  ·  ${CAMPAIGN}"
echo "============================================================"
echo " seeds       : ${SEEDS[*]}   (seed 9 already done twice: 72.81 / 82.69 at S=5)"
echo " training    : ${TRAIN_SCRIPT}"
echo "               args: ${TRAIN_ARGS[*]}"
echo "               cfg_dropout=${CFG_DROPOUT} batch=${TRAIN_BATCH} epochs=${MAX_EPOCHS} fresh"
echo " checkpoint  : ${CKPT_MODE}  <- one rule for every row in every table"
echo -n " sampling    : k=${K}  steps ="
for st in "${STEPS_LIST[@]}"; do echo -n " S=${st} ($((K*st)) NFE)"; done; echo ""
echo " eval        : DiffCSP any-of-k KNOWN-Z on $(basename "$KNOWNZ_CSV"), own candidates"
echo " jobs        : $(( ${#SEEDS[@]} )) trainings, each fanning out to ${#STEPS_LIST[@]} sample+eval chains"
echo " root        : ${ROOT}"
echo "============================================================"
echo " BASELINE TO BEAT: S=5 81.68 +/- 2.63 · S=1 76.30 +/- 2.62 (bestvalid, n=4)."
echo " sd is ~2.6 pts, so a single seed cannot resolve anything smaller."
echo "============================================================"
{
    echo "# campaign ${CAMPAIGN}   $(date)"
    echo "# train=${TRAIN_SCRIPT} args='${TRAIN_ARGS[*]}' cfg_dropout=${CFG_DROPOUT}"
    echo "# k=${K} steps=${STEPS_LIST[*]} max_epochs=${MAX_EPOCHS} ckpt_mode=${CKPT_MODE} csv=${KNOWNZ_CSV}"
    printf "%-6s %-3s %-10s %-10s %-10s  %s\n" "seed" "S" "train_jid" "sample_jid" "eval_jid" "eval_summary"
} >> "$MANIFEST"

train_time_opt=()
[ -n "$TRAIN_TIME" ] && train_time_opt=(--time="$TRAIN_TIME")

for s in "${SEEDS[@]}"; do
    tag="seed${s}"
    LABEL="${CAMPAIGN}_${tag}"
    # CAMPAIGN_LABEL pins hydra.run.dir, so the checkpoint DIRECTORY is
    # deterministic and the dependent sampling job can be submitted up front.
    # The file inside it is chosen at run time (see the wrapper above).
    CKDIR="${TRAIN_RUNS_DIR}/campaign_${LABEL}/checkpoints"

    # 1) TRAIN — fresh (CKPT_PATH_OVERRIDE=null); only the seed differs --------
    #    ONE training per seed; every S value below branches off it.
    TRAIN_JID=$(CAMPAIGN_LABEL="$LABEL" \
                CAMPAIGN_CKPT_REGISTRY="$REG" \
                SEED="$s" \
                CKPT_PATH_OVERRIDE=null \
                MAX_EPOCHS="$MAX_EPOCHS" \
                BATCH_SIZE="$TRAIN_BATCH" \
                CFG_DROPOUT_OVERRIDE="$CFG_DROPOUT" \
                RUN_NAME_PREFIX="${LABEL}_" \
        sbatch --parsable -p "$PART" "${train_time_opt[@]}" \
               --job-name="sdtrain_${tag}" \
               --output="${LOG_DIR}/train_${tag}_%j.out" \
               --error="${LOG_DIR}/train_${tag}_%j.err" \
               "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}")
    TRAIN_JID=${TRAIN_JID%%;*}

    echo ""
    echo "seed=${s}  [${tag}]   train ${TRAIN_JID}"
    echo "  ckpt dir ${CKDIR}  (mode ${CKPT_MODE})"

    for st in "${STEPS_LIST[@]}"; do
        stag="${tag}_s${st}"
        GEN_DIR="${GEN_ROOT}/${stag}"; SEL_DIR="${EVAL_ROOT}/${stag}"

        # 2) SAMPLE — known-Z, formula-only (GT space group never fed) --------
        SAMPLE_JID=$(sbatch --parsable -p "$PART" \
                   --dependency=afterok:${TRAIN_JID} \
                   --gres=gpu:1 \
                   --job-name="sdsample_${stag}" \
                   --output="${LOG_DIR}/sample_${stag}_%j.out" \
                   --error="${LOG_DIR}/sample_${stag}_%j.err" \
                   "$GEN_WRAP" \
                   "$CKDIR" "$CKPT_MODE" "$KNOWNZ_CSV" "$K" "$BATCH" "$st" "$GEN_DIR")
        SAMPLE_JID=${SAMPLE_JID%%;*}

        # 3) EVAL — DiffCSP any-of-k, no relax, no ranker, no SG --------------
        # OWN_CANDIDATES=1: score each material against only its own k samples.
        # Without it the evaluator pools every CIF in the formula folder, so
        # polymorphs sharing a composition are scored against the union (observed
        # up to 135 candidates for k=20), inflating the any-of-k rate. It is also
        # the only setting that EXPOSES a truncated sampling job: pooling lets
        # materials borrow each other's folders and reports n_no_folder=0.
        EVAL_JID=$(DIFFCSP=1 PER_MATERIAL=1 SKIP_SG=1 OWN_CANDIDATES=1 \
            sbatch --parsable -p "$PART" \
                   --dependency=afterok:${SAMPLE_JID} \
                   --job-name="sdeval_${stag}" \
                   --output="${LOG_DIR}/eval_${stag}_%j.out" \
                   --error="${LOG_DIR}/eval_${stag}_%j.err" \
                   "$EVAL_SCRIPT" \
                   "$GEN_DIR" "$GT_DIR" "$KNOWNZ_CSV" "$SEL_DIR" "$K" none)
        EVAL_JID=${EVAL_JID%%;*}

        printf "%-6s %-3s %-10s %-10s %-10s  %s\n" \
            "$s" "$st" "$TRAIN_JID" "$SAMPLE_JID" "$EVAL_JID" \
            "${SEL_DIR}/summary.json" >> "$MANIFEST"

        echo "  S=${st} ($((K*st)) NFE): sample ${SAMPLE_JID} -> eval ${EVAL_JID}  ->  ${SEL_DIR}/summary.json"
    done
done

echo ""
echo "============================================================"
echo "All chains submitted.  Manifest: ${MANIFEST}"
echo ""
echo "Watch:    squeue -u \$USER"
echo "Collect when the eval jobs finish (dir names carry the S value: seed<N>_s<S>):"
echo "  grep -H diffcsp_match_rate  ${EVAL_ROOT}/*/summary.json"
echo "  grep -H diffcsp_mean_rmse   ${EVAL_ROOT}/*/summary.json"
echo "  grep -H n_formulas_total    ${EVAL_ROOT}/*/summary.json   # must be 9046"
echo "  grep -H n_no_folder         ${EVAL_ROOT}/*/summary.json   # must be 0"
echo ""
echo "Per-S summary (mean +/- sd across seeds):"
echo "  for S in ${STEPS_LIST[*]}; do"
echo "    echo -n \"S=\$S  \"; grep -h diffcsp_match_rate ${EVAL_ROOT}/*_s\${S}/summary.json \\"
echo "      | grep -o '[0-9.]*' | awk '{n++;s+=\$1;q+=\$1*\$1} END{m=s/n; print \"n=\"n, \"mean=\"100*m\"%\", \"sd=\"100*sqrt((q-n*m*m)/(n-1))}'"
echo "  done"
echo "============================================================"
