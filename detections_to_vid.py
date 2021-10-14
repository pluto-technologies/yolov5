"""
A small script to generate a mp4 video from a set of captures in a route.

The images must exist in a directory named by `--image-dir`, these can either be the captures downloaded
a Pluto bucket through gsutil or output from yolov5 detections or something else.

To download images through gsutil for a route given by <ROUTE_ID>:
    ```bash
    psql -c "
        select \"ImgName\"
        from \"Captures\" t1
        join \"Positions\" t2 on t1.\"PositionId\" = t2.\"PositionId\"
        where t2.\"RouteId\" = <ROUTE_ID>
        " \
        --csv
    | tail -n +2 \
    | 's/^/gs:\/\/pluto-api-prod-fa69v7gs\//g' \
    | gsutil -m cp -I /data/images/

    ```
    Geneate a video from above export: 
    ```bash
    python detections_to_vid.py \
        --fps 10 \
        --dim 1024 \
        --image-dir /data/images \
        --route-id <ROUTE_ID> \
    ```
"""
import os
import pandas as pd
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from PIL import Image
from PlutoIntegration.lib.connector import PlutoDB

def get_ordered_files(filename):
    with open(filename) as f:
        return [line.strip() if line.strip().endswith('png') else line.strip() + '.png' for line in f]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--image-dir', help="Image directory", default='runs/detect/exp76')
    parser.add_argument('--fps', default=5, type=int, help="Frames per second")
    parser.add_argument('--dim', default=512, type=int, help="Quadratic video resolution")
    parser.add_argument('--route-id', default='a1c14b6f-e1be-46d4-a45a-2e752760b8af', help="Route id")
    parser.add_argument('--from-file', help='Get order of filenames from file instead of DB')

    args = parser.parse_args()

    if args.from_file is not None:
        df = pd.DataFrame(get_ordered_files(args.from_file), columns=['ImgName'])

    else:
        with PlutoDB() as conn:
            conn.execute(f"""
                SELECT "ImgName"
                FROM "Captures" t1
                JOIN "Positions" t2 on t1."PositionId" = t2."PositionId"
                WHERE t2."RouteId" = '{args.route_id}'
                ORDER BY t1."DetectedAt"
            """)
            df = conn.fetch()

    videodims = (args.dim, args.dim)
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video = cv2.VideoWriter(f"{os.path.basename(args.image_dir)}.mp4",fourcc, args.fps,videodims)

    pbar = tqdm(df.itertuples(), total=len(df))

    for i, filename in pbar:
        filename = filename if filename.endswith('.png') else filename + '.png'
        filepath = os.path.join(args.image_dir, filename)
        pbar.refresh()
        if not os.path.isfile(filepath):
            pbar.set_description(f"Missing {filename}")
            continue

        pbar.set_description(f"Adding {filename}")

        img = Image.open(filepath).resize(videodims)

        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    video.release()
