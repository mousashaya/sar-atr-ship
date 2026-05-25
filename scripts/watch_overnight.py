"""
Overnight watcher — emails progress for weighted sampling and TTA runs.

Weighted sampling: emails at n=25, n=100, n=200
TTA: one email when full sweep finishes
"""

import json
import re
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxl vkpr rmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"

ROOT              = Path(__file__).parent.parent
E3_LOG            = ROOT / 'logs' / 'overnight_e3_ablation.log'
ABLATION_LOG      = ROOT / 'logs' / 'overnight_e3_ablation.log'
DINO_LOG          = ROOT / 'logs' / 'prototypical_dinov2' / 'results.json'
MSTAR_LOG         = ROOT / 'logs' / 'overnight_mstar.log'
MSTAR_PROTO_LOG   = ROOT / 'logs' / 'mstar_prototypical' / 'results.json'
MSTAR_TRANSFER_LOG= ROOT / 'logs' / 'mstar_transfer' / 'results.json'

notified = set()


def send(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login(GMAIL_USER, GMAIL_APP_PASS)
        s.send_message(msg)
    print(f"Sent: {subject}")


def check_e3():
    if not E3_LOG.exists():
        return
    text = E3_LOG.read_text()

    for n in [300, 400, 500]:
        key = f'e3_{n}'
        if key in notified:
            continue
        pattern = rf'max_per_group={n}.*?Test accuracy \(15° held-out\): ([\d.]+)%.*?Best val accuracy\s+: ([\d.]+)%'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            test_acc = m.group(1)
            val_acc  = m.group(2)
            subject  = f"E3 n={n} done — test_15°={test_acc}%  val={val_acc}%"
            body     = (
                f"E3 data efficiency n={n} complete.\n\n"
                f"  Test accuracy (held-out 15°): {test_acc}%\n"
                f"  Best val accuracy:            {val_acc}%\n\n"
                f"Previous results:\n"
                f"  n=200: 86.1%\n"
                f"  SOC ceiling (full data): 97.2%\n"
                f"  EOC baseline (15° only): 43.7%\n"
            )
            send(subject, body)
            notified.add(key)


def check_ablation():
    if not ABLATION_LOG.exists():
        return
    text = ABLATION_LOG.read_text()

    for angle_str in ['15+60', '15+30+60']:
        key = f'abl_{angle_str}'
        if key in notified:
            continue
        pattern = rf'Angle Ablation — angles={angle_str}.*?Test accuracy \(15° held-out\): ([\d.]+)%'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            test_acc = m.group(1)
            delta    = float(test_acc) - 43.7
            subject  = f"Ablation {angle_str}° done — {test_acc}% (vs 43.7% baseline, {delta:+.1f}%)"
            body     = (
                f"Angle ablation {angle_str}° complete.\n\n"
                f"  Test accuracy (held-out 15°): {test_acc}%\n"
                f"  vs EOC baseline (15° only):   43.7%\n"
                f"  vs E3 all 4 angles N=200:     86.1%\n"
                f"  Delta vs baseline:            {delta:+.1f}%\n"
            )
            send(subject, body)
            notified.add(key)


def check_dinov2():
    if 'dino_done' in notified:
        return
    if not DINO_LOG.exists():
        return
    try:
        results = json.loads(DINO_LOG.read_text())
    except Exception:
        return
    if not results:
        return
    # Summarize best overall accuracy across strategies
    best = max(r['overall'] for r in results)
    best_r = max(results, key=lambda r: r['overall'])
    delta  = best - 43.7
    subject = f"DINOv2 prototypical done — best={best:.1f}% (vs 43.7% baseline, {delta:+.1f}%)"
    lines = [f"DINOv2 ViT-B/14 prototypical sweep complete (zero ATRNet training).\n"]
    lines.append(f"  Best overall: {best:.1f}%  (n={best_r['n_shot']}, strategy={best_r['strategy']})")
    lines.append(f"  vs EOC baseline (trained 15°-only): 43.7%  delta={delta:+.1f}%")
    lines.append(f"  vs SOC ceiling (trained): 97.2%\n")
    lines.append("Per-N summary:")
    for r in results:
        lines.append(f"  n={r['n_shot']:>4}  {r['strategy']:<8}  {r['overall']:.1f}%")
    send(subject, '\n'.join(lines))
    notified.add('dino_done')


def check_mstar_soc():
    if 'mstar_soc' in notified:
        return
    if not MSTAR_LOG.exists():
        return
    text = MSTAR_LOG.read_text()
    m = re.search(r'Test accuracy \(15°\): ([\d.]+)%', text)
    if not m:
        return
    test_acc = m.group(1)
    m2 = re.search(r'Best val accuracy\s+: ([\d.]+)%', text)
    val_acc = m2.group(1) if m2 else '?'
    subject = f"MSTAR SOC done — test_15°={test_acc}%  (lit: 97-99%)"
    body = (
        f"MSTAR SOC baseline complete (train 17° → test 15°).\n\n"
        f"  Test accuracy (15°): {test_acc}%\n"
        f"  Best val accuracy  : {val_acc}%\n"
        f"  Literature target  : ~97-99%\n"
    )
    send(subject, body)
    notified.add('mstar_soc')


def check_mstar_proto():
    if 'mstar_proto' in notified:
        return
    if not MSTAR_PROTO_LOG.exists():
        return
    try:
        results = json.loads(MSTAR_PROTO_LOG.read_text())
    except Exception:
        return
    if not results:
        return
    best = max(r['accuracy'] for r in results)
    subject = f"MSTAR prototypical done — best={best:.1f}% (MSTAR backbone)"
    lines = ["MSTAR prototypical sweep complete (MSTAR-trained backbone).\n"]
    lines.append(f"  {'N':>6}  {'Accuracy':>10}")
    lines.append(f"  {'-'*20}")
    for r in results:
        lines.append(f"  {r['n_shot']:>6}  {r['accuracy']:>9.1f}%")
    send(subject, '\n'.join(lines))
    notified.add('mstar_proto')


def check_mstar_transfer():
    if 'mstar_transfer' in notified:
        return
    if not MSTAR_TRANSFER_LOG.exists():
        return
    try:
        results = json.loads(MSTAR_TRANSFER_LOG.read_text())
    except Exception:
        return
    if not results:
        return
    best = max(r['accuracy'] for r in results)
    subject = f"Cross-dataset transfer done — ATRNet-STAR→MSTAR best={best:.1f}%"
    lines = ["Cross-dataset transfer complete (ATRNet-STAR backbone, zero MSTAR training).\n"]
    lines.append(f"  {'N':>6}  {'Accuracy':>10}")
    lines.append(f"  {'-'*20}")
    for r in results:
        lines.append(f"  {r['n_shot']:>6}  {r['accuracy']:>9.1f}%")
    lines.append(f"\n  Best: {best:.1f}%  (key paper finding on cross-dataset generalization)")
    send(subject, '\n'.join(lines))
    notified.add('mstar_transfer')


all_done_keys = {'e3_300', 'e3_400', 'e3_500', 'abl_15+60', 'abl_15+30+60', 'dino_done',
                 'mstar_soc', 'mstar_proto', 'mstar_transfer'}

print("Watching E3 extension, angle ablation, DINOv2, and MSTAR runs...")
print(f"  E3/ablation log:  {E3_LOG}")
print(f"  DINOv2 log:       {DINO_LOG}")
print(f"  MSTAR log:        {MSTAR_LOG}")

while True:
    try:
        check_e3()
        check_ablation()
        check_dinov2()
        check_mstar_soc()
        check_mstar_proto()
        check_mstar_transfer()
    except Exception as e:
        print(f"Error: {e}")

    if all_done_keys.issubset(notified):
        send("All overnight runs complete",
             "E3 n=300/400/500, angle ablation, DINOv2 prototypical, and all MSTAR runs finished.")
        print("All done.")
        break

    time.sleep(30)
