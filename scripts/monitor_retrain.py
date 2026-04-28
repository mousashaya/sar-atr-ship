"""
Monitors logs/retrain_fixed_split.log and emails progress per epoch and per model.
Uses a hardcoded model sequence so it never relies on parsing angle from the log.
"""

import re
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

# Load credentials from .env
_env = {}
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            _env[_k.strip()] = _v.strip()

GMAIL_USER     = _env.get('GMAIL_USER', '')
GMAIL_APP_PASS = _env.get('GMAIL_APP_PASSWORD', '')
TO_EMAIL       = _env.get('NOTIFY_EMAIL', GMAIL_USER)
LOG_PATH       = Path(__file__).parent.parent / "logs" / "retrain_fixed_split.log"

# Known training sequence — matches retrain_fixed_split.sh loop order
MODEL_SEQUENCE = [
    ('4class',  15), ('4class',  30), ('4class',  45), ('4class',  60),
    ('21class', 15), ('21class', 30), ('21class', 45), ('21class', 60),
    ('40class', 15), ('40class', 30), ('40class', 45), ('40class', 60),
]
TOTAL = len(MODEL_SEQUENCE)


def send(subject: str, body: str):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASS)
            smtp.send_message(msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")


def read_log() -> str:
    try:
        return LOG_PATH.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ""


def main():
    # Count how many models have already completed (for log re-read on restart)
    models_done  = 0
    seen_epochs  = set()   # (model_idx, epoch_num)

    # Parse split sizes from log segments
    split_re  = re.compile(
        r'Train:\s*([\d,]+)\s+Val:\s*([\d,]+)\s+Test:\s*([\d,]+)'
    )
    epoch_re  = re.compile(
        r'Epoch (\d+)/10\s+loss=([\d.]+)\s+val_acc=([\d.]+)%\s+\((\d+)s\)'
    )
    done_re   = re.compile(r'--- Done config=\w+ angle=\S+ ')
    test_re   = re.compile(r'TEST accuracy:\s*([\d.]+)%')
    val_re    = re.compile(r'VAL accuracy:\s*([\d.]+)%')

    while True:
        if not LOG_PATH.exists():
            time.sleep(30)
            continue

        content    = read_log()
        done_marks = list(done_re.finditer(content))
        models_done = len(done_marks)

        # Current model index
        model_idx = min(models_done, TOTAL - 1)
        config, angle = MODEL_SEQUENCE[model_idx]

        # Find the start of the current model's log segment
        seg_start = done_marks[-1].end() if done_marks else 0
        segment   = content[seg_start:]

        # Split sizes for current model
        sm = split_re.search(segment)
        train_n = sm.group(1) if sm else '?'
        val_n   = sm.group(2) if sm else '?'
        test_n  = sm.group(3) if sm else '?'

        # Per-epoch emails for current model
        for m in epoch_re.finditer(segment):
            epoch   = int(m.group(1))
            loss    = m.group(2)
            val_acc = m.group(3)
            elapsed = int(m.group(4))
            key     = (model_idx, epoch)

            if key in seen_epochs:
                continue
            seen_epochs.add(key)

            epochs_left    = 10 - epoch
            models_queued  = TOTAL - models_done - 1
            secs_remaining = (epochs_left * elapsed) + (models_queued * 10 * elapsed)
            hrs  = secs_remaining // 3600
            mins = (secs_remaining % 3600) // 60

            warning = ""
            if epoch <= 2 and float(val_acc) > 99.0:
                warning = "\n*** WARNING: val_acc >99% in first 2 epochs — possible leakage. ***\n"

            send(
                f"[{config} {angle}deg ep{epoch}/10] val={val_acc}% loss={loss}",
                f"Model {models_done + 1}/{TOTAL}: {config} {angle}deg\n"
                f"Epoch:  {epoch}/10\n"
                f"Loss:   {loss}\n"
                f"Val:    {val_acc}%\n"
                f"Time:   {elapsed}s\n"
                f"{warning}\n"
                f"Split:  Train={train_n}  Val={val_n}  Test={test_n}\n"
                f"Est. remaining: {hrs}h {mins}m\n"
            )

        # Model completion (when a new Done marker appears)
        if models_done > 0:
            prev_idx  = models_done - 1
            prev_cfg, prev_ang = MODEL_SEQUENCE[prev_idx]
            done_key  = (prev_idx, 'done')

            if done_key not in seen_epochs:
                seen_epochs.add(done_key)

                # Find test/val accuracy in the segment just before this Done marker
                prev_seg   = content[:done_marks[-1].start()]
                test_match = test_re.findall(prev_seg)
                val_match  = val_re.findall(prev_seg)
                test_acc   = test_match[-1] if test_match else 'n/a'
                val_final  = val_match[-1]  if val_match  else 'n/a'

                gap_warning = ""
                if test_acc != 'n/a' and val_final != 'n/a':
                    gap = abs(float(val_final) - float(test_acc))
                    if gap > 10:
                        gap_warning = (
                            f"\n*** WARNING: val/test gap = {gap:.1f}pp "
                            f"({val_final}% val vs {test_acc}% test). "
                            f"Possible split issue — check before continuing. ***\n"
                        )

                send(
                    f"[{models_done}/{TOTAL} DONE] {prev_cfg} {prev_ang}deg"
                    f" — test={test_acc}%",
                    f"Model {models_done}/{TOTAL} complete: {prev_cfg} {prev_ang}deg\n\n"
                    f"  Val acc:  {val_final}%\n"
                    f"  Test acc: {test_acc}%\n"
                    f"{gap_warning}\n"
                    f"  {TOTAL - models_done} models remaining.\n"
                )

        if "ALL RETRAINING DONE" in content and models_done >= TOTAL:
            send(
                "ALL 12 MODELS DONE — run eval next",
                "All retraining complete with corrected acquisition-key split.\n\n"
                "Next:\n"
                "  python3 scripts/eval_eoc.py --all-settings\n"
                "  python3 scripts/eval_eoc_depression.py\n"
            )
            break

        time.sleep(20)


if __name__ == '__main__':
    main()
