"""Watch e3_data_efficiency.log and email when n=100 and n=200 finish."""

import re
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxl vkpr rmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"
LOG_PATH       = Path(__file__).parent.parent / 'logs' / 'e3_data_efficiency.log'

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


def check():
    if not LOG_PATH.exists():
        return
    text = LOG_PATH.read_text()

    for n in [100, 200]:
        if n in notified:
            continue
        pattern = rf'max_per_group={n}.*?Test accuracy \(15° held-out\): ([\d.]+)%.*?Best val accuracy\s+: ([\d.]+)%'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            test_acc = m.group(1)
            val_acc  = m.group(2)
            subject  = f"E3 n={n} done — test_15°={test_acc}%  val={val_acc}%"
            body     = (
                f"Data efficiency n={n} complete.\n\n"
                f"  Test accuracy (held-out 15°): {test_acc}%\n"
                f"  Best val accuracy:            {val_acc}%\n\n"
                f"Baselines:\n"
                f"  EOC baseline (15°-only trained): 43.7%\n"
                f"  SOC ceiling  (full data):        97.2%\n"
            )
            send(subject, body)
            notified.add(n)

    if {100, 200}.issubset(notified):
        print("All done. Exiting.")
        raise SystemExit


while True:
    try:
        check()
    except SystemExit:
        break
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(30)
