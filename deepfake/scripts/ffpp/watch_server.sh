#!/bin/bash
# Watches TUM server; relaunches FF++ bulk downloader when it revives. Runs unattended.
cd /Users/Dataghost/Argus_Prime
while true; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -r 0-300 \
    "https://kaldir.vc.in.tum.de/faceforensics/v3/manipulated_sequences/Face2Face/c23/videos/071_054.mp4")
  if [ "$code" = "206" ] || [ "$code" = "200" ]; then
    echo "$(date) server UP ($code), launching downloader" >> /tmp/ffpp_watch.log
    ./.venv/bin/python deepfake/scripts/ffpp/bulk_download.py --workers 24 --passes 6 >> /tmp/ffpp_dl.log 2>&1
    left=$(find data/raw/ffpp/videos -name '*.mp4' | wc -l | tr -d ' ')
    echo "$(date) downloader finished, files=$left" >> /tmp/ffpp_watch.log
    [ "$left" -ge 5000 ] && break
  else
    echo "$(date) server $code" >> /tmp/ffpp_watch.log
  fi
  sleep 600
done
