#!/bin/bash
# EOC 60°-only experiment: train on 60°, test on 15° / 30° / 45°

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
LOG=/Users/mousashaya/Projects/defense/logs/overnight_eoc60.log
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs $CD/logs/eoc_60only

echo "=== EOC 60-only run started $(date) ===" | tee -a $LOG

caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

cd $CD

echo "" | tee -a $LOG
echo "=== Train 60° only → test 15°/30°/45° (40class) $(date) ===" | tee -a $LOG
$PYTHON scripts/train_eoc_60only.py --epochs 10 --config 40class 2>&1 | tee -a $LOG
echo "=== EOC 60-only done $(date) ===" | tee -a $LOG
