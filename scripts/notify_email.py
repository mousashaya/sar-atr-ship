"""
Email notifier — sends epoch-by-epoch status and final results.

Usage:
    python3 scripts/notify_email.py --watch <output_file> --epochs 5 --subject-prefix <prefix>
"""

import argparse
import re
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxl vkpr rmap wmqz".replace(' ', '')
TO_EMAIL       = "mshaya01.ml@gmail.com"


def send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_APP_PASS)
        smtp.send_message(msg)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Email sent: {subject}")


def watch(output_file: str, total_epochs: int, subject_prefix: str, poll_secs: int = 15):
    path          = Path(output_file)
    seen_epochs   = set()
    epoch_times   = []  # elapsed seconds per epoch
    start_time    = None

    print(f"Watching {path}")

    while True:
        if path.exists():
            content = path.read_text()

            # Detect each epoch completion
            for m in re.finditer(
                r'Epoch (\d+)/(\d+)\s+loss=([\d.]+)\s+val_acc=([\d.]+)%\s+\((\d+)s\)',
                content):
                epoch     = int(m.group(1))
                total     = int(m.group(2))
                loss      = float(m.group(3))
                val_acc   = float(m.group(4))
                elapsed   = int(m.group(5))

                if epoch not in seen_epochs:
                    seen_epochs.add(epoch)
                    epoch_times.append(elapsed)
                    if start_time is None:
                        start_time = datetime.now() - timedelta(seconds=elapsed)

                    avg_epoch   = sum(epoch_times) / len(epoch_times)
                    remaining   = (total - epoch) * avg_epoch
                    eta         = datetime.now() + timedelta(seconds=remaining)
                    epochs_left = total - epoch

                    body = (
                        f"Epoch {epoch}/{total} complete\n"
                        f"  Loss:       {loss:.4f}\n"
                        f"  Val acc:    {val_acc:.1f}%  (training distribution — not EOC test)\n"
                        f"  Epoch time: {elapsed}s\n"
                        f"  Remaining:  {epochs_left} epoch(s) + eval\n"
                        f"  ETA:        {eta.strftime('%I:%M %p')}\n"
                    )
                    send_email(f"{subject_prefix} — Epoch {epoch}/{total} ({val_acc:.1f}% val)", body)

            # Detect completion
            if "Results saved" in content:
                lines = content.strip().split('\n')
                body  = '\n'.join(lines[-25:])
                send_email(f"{subject_prefix} — DONE", body)
                return

        time.sleep(poll_secs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch',          required=True)
    parser.add_argument('--epochs',         type=int, default=5)
    parser.add_argument('--subject-prefix', default='Training')
    args = parser.parse_args()
    watch(args.watch, args.epochs, args.subject_prefix)


if __name__ == '__main__':
    main()
