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
import math
from dataclasses import dataclass

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

trusts = {
  "A20":{
    "name": "plate",
    "threshold": 30,
    "trusted": 30
  },
  "D40": {
    "name": "hole",
    "threshold": 40,
    "trusted": 100
  },
  "D50": {
    "name": "depression",
    "threshold": 50,
    "trusted": 50
  },
  "A10": {
    "name": "face",
    "threshold": 30,
    "trusted": 30
  },
  "P20": {
    "name": "lane marking",
    "threshold": 10,
    "trusted": 100
  },
  "D20": {
    "name": "crocodile crack",
    "threshold": 10,
    "trusted": 100
  },
  "D10": {
    "name": "crack",
    "threshold": 20,
    "trusted": 100
  },
  "R30": {
    "name": "crack seal",
    "threshold": 20,
    "trusted": 70
  },
  "S10": {
    "name": "permanent sign",
    "threshold": 40,
    "trusted": 40
  },
  "D30": {
    "name": "raveling",
    "threshold": 20,
    "trusted": 100
  },
  "R10": {
    "name": "area patch",
    "threshold": 65,
    "trusted": 65
  },
  "M10": {
    "name": "manhole",
    "threshold": 60,
    "trusted": 60
  },
  "M20": {
    "name": "drain",
    "threshold": 70,
    "trusted": 70
  },
  "S20": {
    "name": "temporary sign",
    "threshold": 30,
    "trusted": 30
  },
  "R20": {
    "name": "spot patch",
    "trusted": 50,
    "threshold": 50
  },
  "S30": {
    "name": "sign back",
    "threshold": 30,
    "trusted": 30
  },
  "S40": {
    "name": "bollard",
    "threshold": 65,
    "trusted": 65
  }
}


thresholds =  {
    "plate": .3,
    "hole": .3,
    "depression": .5,
    "face": .3,
    "lane marking": .1,
    "crocodile crack": .05,
    #"crocodile crack": .2,
    # "crack": .15,
    "crack": .15,
    "crack seal": .2,
    "permanent sign": .4,
    "raveling": .2,
    "area patch": .65,
    "manhole": .6,
    "drain": .7,
    "temporary sign": .3,
    "spot patch": .5,
    "sign back": .3,
    "bollard": .65,
}


@dataclass
class Point:
    x: float
    y: float


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

name_to_trust = { t['name']: t['trusted'] for t in trusts.values() }

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

def get_command(command=None, municipalities=None, routeId=None, routeFromCapture=None, captureId=None):
        if command is not None:
            return command
        elif captureId is not None:
            captureId = captureId.split('/')[-1]
            return f"""
            SELECT *, t1.xmin from "Captures" t1
            LEFT JOIN "Calibrations" t2 ON t1."CalibrationId" = t2."CalibrationId"
            WHERE "CaptureId" = '{captureId}'
            """
        elif routeId is not None:
            return f"""
            SELECT *, t1.xmin FROM "Captures" t1
            LEFT JOIN "Calibrations" t2 ON t1."CalibrationId" = t2."CalibrationId"
            JOIN "Positions" t3 ON t1."PositionId" = t3."PositionId"
            WHERE t3."RouteId" = '{routeId}'
            AND "Representing" AND NOT "Reviewed"
            """
        elif routeFromCapture is not None:
            captureId = routeFromCapture.split('/')[-1]
            return f"""
            SELECT *, t1.xmin FROM "Captures" t1
            LEFT JOIN "Calibrations" t2 ON t1."CalibrationId" = t2."CalibrationId"
            JOIN "Positions" t3 ON t1."PositionId" = t3."PositionId"
            WHERE t3."RouteId" IN (
                SELECT "RouteId"
                FROM "Positions" t1
                JOIN "Captures" t2 on t1."PositionId" = t2."PositionId"
                WHERE t2."CaptureId" = '{captureId}'
            )
            AND "Representing" AND NOT "Reviewed"
            """
        elif municipalities is not None:
            if type(municipalities) == str:
                municipalities = [municipalities]
            clause = "'" + "', '".join(municipalities) + "'"

            return f"""
            SELECT t1.*, t1.xmin, t3.* FROM "Captures" t1
            JOIN "Municipalities" t2 on t1."MunicipalityId" = t2."Id"
            LEFT JOIN "Calibrations" t3 on t1."CalibrationId" = t3."CalibrationId"
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
        RETURNING "AnnotationId";
        """)
        print(f"Removed {len(conn.fetch())} existing annotations")
        conn.execute(f"""
        with captures AS (
            {command}
        )
        UPDATE "Captures"
        set "Trusted" = False

        where "CaptureId" IN (SELECT "CaptureId" FROM captures)
        returning "CaptureId";
        """)
        print(f"Reset trusts for {len(conn.fetch())} captures")
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

def get_capture_url(env, capture):
    prefix = '' if env == "prod" else f"{env}."
    return f"https://{prefix}plutomap.com/map/capture/{capture.CaptureId}"


def process_predictions(env, capture, pred, img0, store_asymmetric=False):
    detections = []
    s = ''
    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        url = get_capture_url(env, capture)
        s += f"🔗 {url} 👉 "

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            detections.append((names[c], *map(int, [cls, *xyxy]), *map(float, (conf,)))) # format

            """
            if store_asymmetric and obj_conf < .20 and cls_conf > .80:
                ## Special case for hunting crocs and cracks with low obj conf and high cls conf:
                with open('crack_croc_low_high.csv', 'a') as f:
                    f.write(f""" "{names[c]}", "{conf}", "{obj_conf}", "{cls_conf}", "{url}" """.strip())
                    f.write("\n")
            """


    return detections, s + '👈'


def create_annotations(url, headers, capture, detections, stats={}):
    url = url if url.endswith('batch') else os.path.join(url, 'api/v1/annotations/batch')
    annotations = []
    for name, cls, x1, y1, x2, y2, conf in detections:
        annotation = {
            "annotationId": str(uuid4()),
            "className" : name_to_cls[name.lower()],
            "severe": False,
            "isFixed": False,
            "confidence": conf,
            "captureId": capture.CaptureId,
            "x1": x1, "x2": x2, "y1": y1, "y2": y2,
        }

        try:
            vanishingPoint = getVanishingPoint(capture.x1,
                                               capture.x2,
                                               capture.x3,
                                               capture.x4,
                                               capture.y1,
                                               capture.y2,
                                               capture.y2,
                                               capture.y1)

            annotation = getAnnotationMeasure(annotation, capture, vanishingPoint)
        except AttributeError:
            # There was no calibration
            pass

        annotations.append(annotation)
        if name in stats:
            stats[name] += 1
        else:
            stats[name] = 1

    # requests.post(url, headers=headers, json=annotations)
    return grequests.post(url, headers=headers, json=annotations), stats


def euclid(p1: Point, p2: Point):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def translate(point: Point, angle: float, unit: float) -> Point:
  x, y = point.x, point.y

  x += unit*math.cos(angle)
  y += unit*math.sin(angle)

  return Point(x=x, y=y)


def getVanishingPoint(x1, x2, x3, x4, y1, y2, y3, y4):
    """
    See: https://github.com/pluto-technologies/Pluto-Technology/blob/8a87003816c61c1d2516597e439bdc631d6afa6d/webapp/src/helpers/calibrationHelper.ts#L100-L139
    """
    upper_left = Point(x1, y1)
    lower_left = Point(x2, y2)
    lower_right = Point(x3, y3)
    upper_right = Point(x4, y4)

    a = euclid(lower_left, lower_right)

    ## Triangle 1:
    b1 = euclid(upper_left, lower_left)
    c1 = euclid(upper_left, lower_right)

    angle1 = math.acos(
        (a**2 + b1**2 - c1**2) / (2 * a * b1)
    )

    ## Triangle 2:
    b2 = euclid(lower_right, upper_right)
    c2 = euclid(upper_right, lower_left)
    ## angle2 = math.acos(b2**2 + c2**2 - a**2) / (2 * b2 * c2))
    angle2 = math.acos(
        (a**2 + b2**2 - c2**2) / (2 * a * b2)
    )

    ## Compute point of vanishing
    a3 = x1 - x2
    b3 = b1
    c3 = y1 - y2
    ## The angle relates to the orientation of the image, and not just the angle compared to the other sides of the mat:
    angle = (180 * math.pi/180) - math.acos(
        (a3**2 + b3**2 - c3**2)  / (2 * a3 * b3)
    )
    distance = a * (
        math.sin(angle2) / (
            math.sin(angle1) * math.cos(angle2) + math.sin(angle2)*math.cos(angle1)
        )
    )
    return translate(lower_left, angle, -1 * distance)


def getPointWithin(point: Point, vanishingPoint: Point) -> Point:
    return Point(
        x=point.x,
        y=point.y if point.y > vanishingPoint.y else vanishingPoint.y
    )


def getAnnotationMeasure(annotation, calibration=None, vanishingPoint=None):
    if calibration == None: return annotation
    ## Transform the undistorted point by the perspective matrix.
    tPoint = lambda p: transformPoint(p, calibration)

    if (vanishingPoint != None and annotation['y1'] <= vanishingPoint.y and annotation['y2'] <= vanishingPoint.y ):
        annotation['bottomMeasure'] = 0
        annotation['leftMeasure'] = 0
        return annotation


    checkVanishing = lambda p: p if vanishingPoint == None else getPointWithin(p, vanishingPoint)

    upper_left = tPoint(checkVanishing(Point(annotation['x1'], annotation['y1'])))
    lower_left = tPoint(checkVanishing(Point(annotation['x1'], annotation['y2'])))
    lower_right = tPoint(checkVanishing(Point(annotation['x2'], annotation['y2'])))

    bottom = euclid(lower_left, lower_right)
    left = euclid(lower_left, upper_left)
    annotation['bottomMeasure'] = bottom
    annotation['leftMeasure'] = left
    return annotation


def transformPoint(p: Point, calibration) -> Point:
    pm = np.asarray([
        [calibration.pm_0_0, calibration.pm_0_1, calibration.pm_0_2],
        [calibration.pm_1_0, calibration.pm_1_1, calibration.pm_1_2],
        [calibration.pm_2_0, calibration.pm_2_1, calibration.pm_2_2]
    ])
    a, b, c = np.dot(pm, np.array([p.x, p.y, 1]))
    return Point( x = a/c, y = b/c )


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


def set_trusted(url, headers, capture, detections):
    url = os.path.join(url, 'api/v1/captures?setsReviewed=false')

    trust_all = True # Implicitly meaning that we trusts captures without detections too
    for name, cls, x1, y1, x2, y2, conf in detections:
        trust_all &= name_to_trust[name] < conf * 100

    if trust_all:
        with PlutoDB() as conn:
            conn.execute(f"""
                UPDATE "Captures"
                set "Trusted" = {trust_all}
                WHERE "CaptureId" = '{capture.CaptureId}'
                RETURNING "CaptureId"
            """)
            # print(f"Updated {len(conn.fetch())} capture as trusted")
            conn.connection.commit()

    return trust_all


################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='dev', help="Environment of which API to use")
    parser.add_argument('--user', help="Pluto username, if set overrides PLUSER")
    parser.add_argument('--password', help="Pluto password, if set overrides PLPASSWORD")
    parser.add_argument('-c', '--command', help="SQL command to select captures based of")
    parser.add_argument('-m', '--municipalities', nargs='*', help="Only from these municipalities")
    parser.add_argument('--capture', help='Only on this capture, captureid or capture link')
    parser.add_argument('--route-from-capture', help='Entire route based on provided capture id or link')
    parser.add_argument('--route-id', help='Entire route based on provided route id')
    parser.add_argument('--weights', default='runs/train/exp52/weights/best.pt')
    #parser.add_argument('--weights', default='runs/train/exp52/weights/best.torchscript.ptl')
    parser.add_argument('--iou-thres', type=float, default=0.2)
    parser.add_argument('--max-det', type=int, default=50)
    parser.add_argument('--remove-existing', action='store_true', help='If set, removes existing annotaitons for the involved captures')
    parser.add_argument('--list-municipalities', action='store_true', help='List municipalities and exit, takes precedence over other options')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without making changes')
    parser.add_argument('--num-reqs', default=40, type=int, help="Number of requests to send at a time, default: 40)")
    parser.add_argument('--store-asymmetric-conf', action="store_true", help="Store CSV of image URLs with very asymmetric class and object confidences")
    parser.add_argument('--refresh', action='store_true', help="If set, will refresh materialized view if any captures were trusted")

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
                          routeId=args.route_id,
                          routeFromCapture=args.route_from_capture,
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
    trusted = 0
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
                                   max_det=args.max_det,
                                   cls_exclusive=torch.Tensor(cls_exclusvies))
        dt[2] += time_sync() - t3


        detections, s = process_predictions(args.env, capture, pred, img0.copy(), args.store_asymmetric_conf)
        total += len(detections)
        if len(detections) and not args.dry_run:
            pbar.set_description(s)
            pbar.refresh()
            req, stats = create_annotations(url, get_headers(token), capture, detections, stats=stats)
            reqs.append(req)
            if len(reqs) > args.num_reqs:
                grequests.map(reqs)
                reqs = []

        trusted_capture = set_trusted(url, get_headers(token), capture, detections)
        if trusted_capture:
            s += ' [🔒 TRUSTED 🔒]'
            pbar.set_description(s)
            pbar.refresh()
            trusted += 1

        if False and count > 200: break

    grequests.map(reqs)
    pbar.refresh()
    pbar.close()
    print()

    print(f"Made total of {total} annotations:")
    print(json.dumps(stats, sort_keys=True, indent=2))

    print(f"Trusting {trusted} of {count} captures [{trusted/count * 100}%]")
    if args.refresh and trusted > 0:
        print(f"Refreshing materialized view since {trusted} captures were trusted")
        with PlutoDB() as c:
            c.execute("refresh materialized view capturelayer")
            c.connection.commit()
