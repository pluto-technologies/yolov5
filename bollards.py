#!/usr/bin/env python3
# Pluto 🛣

import os
import sys
import subprocess

from PlutoIntegration.lib.connector import PlutoDB
from PlutoIntegration.queries import captures

#IMG_DIR = '/data/images/'
IMG_DIR = '/tmp/pluto/images/'
weights = '/home/kalk/yolov5/runs/train/exp24/weights/best.pt'

with PlutoDB() as c:
    df = captures.getReviewedCaptures(c)
    df['images'] = df.ImgName.apply(lambda s: os.path.join(IMG_DIR, s + '.png'))

    source = '/tmp/images.txt'
    df[['images']].to_csv(source, header=False, index=False)

    cmd = f"""
    python detect.py --weights {weights} --source {source} --imgsz 1024 --nosave --save-txt
    """

    print(cmd)

