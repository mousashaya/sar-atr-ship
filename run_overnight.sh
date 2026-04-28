#!/bin/bash
# Overnight run: restricted EOC training + eval for all 4 non-depression conditions
# Uses ConvNeXt-Small (paper's baseline) for apples-to-apples comparison
# Model naming: convnext_{model_type}_40class_{cond_tag}_angle{A}_v1
# Eval config:  --config {model_type}_40class_{cond_tag}

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
LOG=/Users/mousashaya/Projects/defense/logs/overnight_run.log
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs

echo "=== Overnight run started $(date) ===" | tee -a $LOG

# Wait for hierarchy eval to finish (if still running)
HIER_PID=$(pgrep -f eval_eoc_hierarchy_full)
if [ -n "$HIER_PID" ]; then
    echo "=== Waiting for hierarchy eval (PID $HIER_PID) ===" | tee -a $LOG
    while kill -0 $HIER_PID 2>/dev/null; do sleep 30; done
    echo "=== Hierarchy eval done ===" | tee -a $LOG
fi

cd $CD

# ── Step 1: Train restricted classifiers with ConvNeXt-Small ─────────────────
# Each condition trains 4 angle-specific 40-class classifiers (15/30/45/60°)
# EOC-Depression is EXCLUDED: it needs 15°-only training, handled separately

declare -A COND_TO_SETTING=(
    ["EOC-Scene"]="EOC_scene"
    ["EOC-Azimuth"]="EOC_azimuth"
    ["EOC-Band"]="EOC_band"
    ["EOC-Pol"]="EOC_pol"
)

for CONDITION in "EOC-Scene" "EOC-Azimuth" "EOC-Band" "EOC-Pol"; do
    COND_TAG=$(echo $CONDITION | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    echo "" | tee -a $LOG
    echo "=== Training $CONDITION (ConvNeXt-Small, restricted) $(date) ===" | tee -a $LOG
    $PYTHON scripts/train_eoc_restricted.py \
        --condition $CONDITION \
        --model-type small \
        --name v1 \
        --epochs 10 \
        --batch-size 32 \
        --num-workers 4 2>&1 | tee -a $LOG
    echo "=== $CONDITION training done $(date) ===" | tee -a $LOG
done

# ── Step 2: Eval each condition with angle routing ────────────────────────────
for CONDITION in "EOC-Scene" "EOC-Azimuth" "EOC-Band" "EOC-Pol"; do
    COND_TAG=$(echo $CONDITION | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    SETTING=${COND_TO_SETTING[$CONDITION]}
    CONFIG="small_40class_${COND_TAG}"
    echo "" | tee -a $LOG
    echo "=== Evaluating $SETTING (config=$CONFIG) $(date) ===" | tee -a $LOG
    $PYTHON scripts/eval_eoc.py \
        --setting $SETTING \
        --config $CONFIG 2>&1 | tee -a $LOG
    echo "=== $SETTING eval done ===" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== ALL DONE $(date) ===" | tee -a $LOG
echo "=== Results are in logs/overnight_run.log ===" | tee -a $LOG
