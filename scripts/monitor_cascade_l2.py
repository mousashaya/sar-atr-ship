"""
Monitors logs/train_cascade_level2.log and emails on each model completion.
Designed to work even when attached mid-run.
"""

import re
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

# Load credentials from .env
env_path = Path(__file__).parent.parent / '.env'
env = {}
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()

GMAIL_USER = env.get('GMAIL_USER', '')
GMAIL_PASS = env.get('GMAIL_APP_PASSWORD', '')
TO_EMAIL   = env.get('NOTIFY_EMAIL', GMAIL_USER)
LOG_PATH   = Path(__file__).parent.parent / 'logs' / 'train_cascade_level2.log'

TOTAL = 16  # 4 classes x 4 angles


def send(subject: str, body: str):
    if not GMAIL_USER or not GMAIL_PASS:
        print(f"[{now()}] Email skipped — no credentials")
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = GMAIL_USER
        msg['To']      = TO_EMAIL
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_PASS)
            smtp.send_message(msg)
        print(f"[{now()}] Sent: {subject}")
    except Exception as e:
        print(f"[{now()}] Email failed: {e}")


def now():
    return datetime.now().strftime('%H:%M:%S')


def read_log():
    try:
        return LOG_PATH.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ""


def main():
    seen_done = set()
    done_re   = re.compile(r'--- Done L2 parent=(\S+) angle=(\d+)deg')
    test_re   = re.compile(r'TEST accuracy:\s*([\d.]+)%')
    val_re    = re.compile(r'VAL accuracy:\s*([\d.]+)%')

    print(f"[{now()}] Monitor started — watching {LOG_PATH}")
    send("Cascade L2 monitor started", f"Watching {TOTAL} models in {LOG_PATH}")

    while True:
        if not LOG_PATH.exists():
            time.sleep(15)
            continue

        content   = read_log()
        done_list = list(done_re.finditer(content))
        n_done    = len(done_list)

        for i, m in enumerate(done_list):
            parent = m.group(1)
            angle  = m.group(2)
            key    = (parent, angle)
            if key in seen_done:
                continue
            seen_done.add(key)

            # Find test/val in the segment before this Done marker
            seg      = content[:m.start()]
            # Get the segment since the previous Done marker
            prev_end = done_list[i-1].end() if i > 0 else 0
            seg      = content[prev_end:m.start()]

            test_matches = test_re.findall(seg)
            val_matches  = val_re.findall(seg)
            test_acc = test_matches[-1] if test_matches else 'n/a'
            val_acc  = val_matches[-1]  if val_matches  else 'n/a'

            gap_warning = ""
            if test_acc != 'n/a' and val_acc != 'n/a':
                gap = abs(float(val_acc) - float(test_acc))
                if gap > 8:
                    gap_warning = f"\n⚠ Val/test gap = {gap:.1f}pp — possible overfitting or hard subclass boundary."

            remaining = TOTAL - n_done
            send(
                f"[L2 {i+1}/{TOTAL}] {parent}/{angle}° — test={test_acc}%",
                f"Level 2 model {i+1}/{TOTAL} complete\n\n"
                f"  Parent class: {parent}\n"
                f"  Angle:        {angle}°\n"
                f"  Val acc:      {val_acc}%\n"
                f"  Test acc:     {test_acc}%\n"
                f"{gap_warning}\n"
                f"  {remaining} models remaining.\n"
            )

        if "ALL LEVEL 2 DONE" in content and n_done >= TOTAL:
            # Build summary table
            lines = ["All 16 Level 2 models complete!\n", f"{'Model':<25} {'Test':>6}", "-"*33]
            for m in done_list:
                seg_start = done_list[done_list.index(m)-1].end() if done_list.index(m) > 0 else 0
                seg = content[seg_start:m.start()]
                t = test_re.findall(seg)
                lines.append(f"{m.group(1)}/{m.group(2)}°{'':<15} {t[-1]+'%' if t else 'n/a':>6}")
            send("ALL Level 2 DONE — ready to review", '\n'.join(lines))
            break

        time.sleep(20)


if __name__ == '__main__':
    main()
