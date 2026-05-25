#!/bin/bash
PASS=$(awk '/earthdata/{getline; getline; print $2}' ~/.netrc)
wget --http-user=shayaml --http-password="$PASS" --auth-no-challenge \
  --load-cookies /tmp/urs_cookies --save-cookies /tmp/urs_cookies \
  --keep-session-cookies --max-redirect=10 \
  -O /tmp/test_slc.zip \
  "https://datapool.asf.alaska.edu/SLC/SA/S1A_IW_SLC__1SDV_20240531T122429_20240531T122458_054115_069496_B458.zip"
echo "Exit code: $?"
ls -lh /tmp/test_slc.zip 2>/dev/null
