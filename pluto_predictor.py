#!/usr/bin/env python3

import grequests
import requests
import argparse
import json
import os
import sys
from detect import run as detect
import numpy as np
from io import BytesIO
import torch
from tqdm import tqdm
from PIL import Image
from PlutoIntegration.lib.connector import PlutoDB
from uuid import uuid4

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

thresholds =  {
    "plate": .3,
    "hole": .3,
    "depression": .5,
    "face": .3,
    "lane marking": .1,
    #"crocodile crack": .15,
    "crocodile crack": .2,
    "crack": .15,
    "crack seal": .2,
    "permanent sign": .4,
    "raveling": .2,
    "area patch": .6,
    "manhole": .6,
    "drain": .7,
    "temporary sign": .3,
    "spot patch": .3,
    "sign back": .3,
    "bollard": .65,
}

# The numbers has no intrinsic meaning, but those classes with the same number will
# be treated as mutually exclusive.
cls_exclusvies = [
    0, # "plate": .3,
    1, # "hole": .3,
    2, # "depression": .5,
    3, #, "face": .3,
    4, #" "lane marking": .2,
    5, # "crocodile crack": .1,
    5, # "crack": .15,
    7, # "crack seal": .2,
    8, # "permanent sign": .4,
    9, # "raveling": .2,
    10, #"area patch": .6,
    11, # "manhole": .6,
    12, # "drain": .7,
    13, # "temporary sign": .3,
    14, # "spot patch": .2,
    15, #, "sign back": .3,
    13, # "bollard": .65,
]
## torch.index_select(torch.Tensor([8,9,10,11]).to(x.device), 0, x[:, 5].int())

names = [k for k, _ in thresholds.items()]

pluto_to_name = {
0 :"plate",
1 :"hole",
2 :"crack",
3 :"depression",
4 :"crosswalk",
5 :"crack",
6 :"face",
7 :"crack",
8 :"lane marking",
9 :"crocodile crack",
10:"crack",
11:"pothole",
12:"curb Damage",
13:"crack seal",
14:"permanent sign",
15:"raveling",
16:"area patch",
17:"manhole",
18:"drain",
19:"skid mark",
20:"hydrant",
21:"sidewalk tile crack",
22:"Vegetation from below",
23:"Bleeding",
24:"Scaffolding",
25:"Container",
26:"Wooden stick",
27:"Oil spill",
28:"Temporary sign",
29:"spot Patch",
30:"Sign back",
31:"Bollard",
32:"Rutting",
33:"Edge deterioration",
}
name_to_cls = { v.lower(): k for k, v in pluto_to_name.items() }

def get_headers(auth_token = None):
    headers = {} if auth_token is None else {
        "Authorization": f"Bearer {auth_token}"
    }
    headers.update({
        "Content-Type": "application/json",
    })
    return headers

def authenticate(url, user, pwd):
    url = url if url.endswith('authenticate') else os.path.join(url, 'api/v1/auth/authenticate')
    return requests.post(url,
                         json={"username": user, "password": pwd},
                         headers=get_headers()
     ).json().get('authorization', {}).get('accessToken')

def get_command(command=None, municipalities=None, captureId=None):
        if command is not None:
            return command
        elif captureId is not None:
            captureId = captureId.split('/')[-1]
            return f"""
            SELECT * from "Captures"
            WHERE "CaptureId" = '{captureId}'
            """
        elif municipalities is not None:
            if type(municipalities) == str:
                municipalities = [municipalities]
            clause = "'" + "', '".join(municipalities) + "'"

            return f"""
            SELECT t1.* FROM "Captures" t1
            JOIN "Municipalities" t2 on t1."MunicipalityId" = t2."Id"
            WHERE t2."Municipality" = ANY(ARRAY[{clause}])
            AND "Representing" AND NOT "Reviewed"
            ORDER BY "CreatedAt" DESC
            """

        raise NotImplementedError("Option not supportted")


def get_captures(command):
    with PlutoDB() as conn:
        conn.execute(command)
        return conn.fetch()


def remove_existing_annotations(command):
    with PlutoDB() as conn:
        conn.execute(f"""
        WITH captures AS (
            {command}
        ),
        annotations as (
            DELETE FROM "Annotations"
            WHERE "CaptureId" IN (SELECT "CaptureId" from captures)
            RETURNING "AnnotationId"
        )
        DELETE FROM "AnnotationHistory" where "AnnotationId" IN (SELECT * FROM annotations)
        RETURNING *;
        """)
        print(f"Removed {len(conn.fetch())} existing annotations")
        conn.connection.commit()


def get_images(url, df, headers, num_reqs=20):
    """
    Yields Captures and PIL.Image objects from Pluto API

    Requires df to include a column named `CaptureId` for each image to fetch.

    """
    creqs = [] # Capture requests
    captures = []
    pbar = tqdm(df.itertuples(), total=len(df), unit='captures', colour='GREEN', leave=False)
    for capture in pbar:
        endpoint = os.path.join(url, 'api/v1/captures', capture.CaptureId)
        creqs.append(grequests.get(endpoint, headers=headers))
        captures.append(capture)
        if len(creqs) > num_reqs:
            imgreqs = [
                grequests.get(img.json().get('imgPath'))
                for img in grequests.map(creqs)
            ]
            for c, imgRes in zip(captures, grequests.map(imgreqs)):
                image = Image.open(BytesIO(imgRes.content))
                yield c, image
            creqs = []
            captures = []

    imgreqs = [
        grequests.get(img.json().get('imgPath'))
        for img in grequests.map(creqs)
    ]
    # Empty any outstanding requests
    for c, imgRes in zip(captures, grequests.map(imgreqs)):
        image = Image.open(BytesIO(imgRes.content))
        yield c, image

    pbar.refresh()
    pbar.close()


def to_yolov5_dataloader(images_gen, stride, img_size=1024):
    """
    Mimicks the output of YOLOv5's ImageLoader class
    but instead of reading from a filepath, the images are loaded from Pluto API
    """
    for capture, image in images_gen:

        # To numpy:
        img0 = np.array(image)[:, :, ::-1].copy()

        # Padded resize
        img = letterbox(img0, img_size, stride=stride, auto=False)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        yield capture, f"/tmp/{capture.ImgName}.png", img, img0


def get_model(weights, device):
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    return model, stride

def prepare_img(device, img):
    img = torch.from_numpy(img).to(device)
    img.float()
    img = img / 255.0
    if len(img.shape) == 3:
        img = img[None]
    return img


def process_predictions(env, capture, pred, img0):
    detections = []
    s = ''
    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        #s += '%gx%g ' % img.shape[2:]  # print string
        prefix = '' if env == "prod" else f"{env}."
        url = f"https://{prefix}plutomap.com/map/capture/{capture.CaptureId}"
        s += f"🔗 {url} 👉 "

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            detections.append((names[c], *map(int, [cls, *xyxy]), float(conf))) # format

    return detections, s + '👈'



def create_annotations(url, headers, capture, detections, stats={}):
    url = url if url.endswith('batch') else os.path.join(url, 'api/v1/annotations/batch')
    annotations = []
    for name, cls, x1, y1, x2, y2, conf in detections:
        annotations.append({
            "annotationId": str(uuid4()),
            "className" : name_to_cls[name.lower()],
            "severe": False,
            "isFixed": False,
            "confidence": conf,
            "captureId": capture.CaptureId,
            "x1": x1, "x2": x2, "y1": y1, "y2": y2
        })
        if name in stats:
            stats[name] += 1
        else:
            stats[name] = 1

    # requests.post(url, headers=headers, json=annotations)
    return grequests.post(url, headers=headers, json=annotations), stats


def threshold_annotation(preds, device):
    thres = torch.Tensor([v for k, v in thresholds.items()])
    nc = preds.shape[2] - 5
    # xc = preds[..., 4] > .15 # initial candidates
    xc = preds[..., 4] > .01 # initial candidates

    for xi, x in enumerate(preds):
        x = x[xc[xi]]

        # Compute conf
        confs = x[:, 5:] * x[:, 4:5] > thres.to(device)  # conf = obj_conf * cls_conf

        # Truncate all other non-satisfying thresholds:
        x[:, 5:] *= confs

        candidates = x[confs.any(1)] # These are the ones satisfying being above individual thresholds

        return candidates[None, :, :]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='dev', help="Environment of which API to use")
    parser.add_argument('--user', help="Pluto username, if set overrides PLUSER")
    parser.add_argument('--password', help="Pluto password, if set overrides PLPASSWORD")
    parser.add_argument('-c', '--command', help="SQL command to select captures based of")
    parser.add_argument('-m', '--municipalities', nargs='*', help="Only from these municipalities")
    parser.add_argument('--capture', help='Only on this capture, captureid or capture link')
    parser.add_argument('--weights', default='runs/train/exp52/weights/best.pt')
    #parser.add_argument('--weights', default='runs/train/exp52/weights/best.torchscript.ptl')
    parser.add_argument('--iou-thres', type=float, default=0.2)
    parser.add_argument('--max-det', type=int, default=50)
    parser.add_argument('--remove-existing', action='store_true', help='If set, removes existing annotaitons for the involved captures')
    parser.add_argument('--list-municipalities', action='store_true', help='List municipalities and exit, takes precedence over other options')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without making changes')
    parser.add_argument('--num-reqs', default=40, type=int, help="Number of requests to send at a time, default: 40)")

    # Setup
    args = parser.parse_args()

    if args.list_municipalities:
        with PlutoDB() as c:
            c.execute("""SELECT "Municipality" FROM "Municipalities";""")
            print(c.fetch())
        sys.exit(0)

    url = f"https://{args.env}.api.plutomap.com"
    user = args.user or os.environ.get('PLUSER')
    password = args.password or os.environ.get('PLPASSWORD')
    assert user is not None and password is not None, "No Pluto credentials provided"
    token = authenticate(url, user, password)

    # Fetch images and prepare db
    command = get_command(command=args.command,
                          municipalities=args.municipalities,
                          captureId=args.capture)

    captures_df = get_captures(command)
    print(f"Fetched {len(captures_df)} captures")
    if args.remove_existing and not args.dry_run:
        remove_existing_annotations(command)

    # Make predecitions
    device = select_device('')
    model, stride = get_model(args.weights, device)

    images_gen = get_images(url, captures_df, get_headers(token), num_reqs=args.num_reqs)

    pbar = tqdm(to_yolov5_dataloader(images_gen, stride))

    dt, seen = [0.0, 0.0, 0.0], 0
    stats = {}
    total = 0
    count = 0
    reqs = []
    for capture, p, img, img0 in pbar:
        count += 1
        t1 = time_sync()

        img = prepare_img(device, img)

        t2  = time_sync()
        dt[0] += t2 - t1

        pred = model(img, augment=False, visualize=False)[0]

        t3 = time_sync()
        dt[1] += t3 - t2

        pred = threshold_annotation(pred, device)

        # NMS
        pred = non_max_suppression(pred,
                                   0.001, # Thresholds have already been applied - only do NMS
                                   args.iou_thres,
                                   None,
                                   False,
                                   max_det=args.max_det)
        dt[2] += time_sync() - t3


        detections, s = process_predictions(args.env, capture, pred, img0.copy())
        total += len(detections)
        if len(detections) and not args.dry_run:
            pbar.set_description(s)
            req, stats = create_annotations(url, get_headers(token), capture, detections, stats=stats)
            reqs.append(req)
            if len(reqs) > args.num_reqs:
                grequests.map(reqs)
                reqs = []

        if False and count > 200: break

    grequests.map(reqs)
    pbar.refresh()
    pbar.close()
    print()

    print(f"Made total of {total} annotations:")
    print(json.dumps(stats, sort_keys=True, indent=2))
