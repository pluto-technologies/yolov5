#!/usr/bin/env python3
# 🛣
#
##      Column     |         Type         | Collation | Nullable | Default
##  ---------------+----------------------+-----------+----------+---------
##   AnnotationId  | uuid                 |           | not null |
##   ClassName     | integer              |           | not null |
##   Severe        | boolean              |           | not null |
##   IsFixed       | boolean              |           | not null |
##   Confidence    | double precision     |           | not null |
##   X1            | integer              |           | not null |
##   X2            | integer              |           | not null |
##   Y1            | integer              |           | not null |
##   Y2            | integer              |           | not null |
##   CaptureId     | uuid                 |           | not null |
##   bottomMeasure | real                 |           | not null | 0
##   leftMeasure   | real                 |           | not null | 0
##   Location      | geometry(Point,4326) |           |          |
##   CrackType     | text                 |           |          |
##   RoadSign      | text                 |           |          |

from datetime import datetime
import os
import uuid

from PlutoIntegration.lib.connector import PlutoDB
from PlutoIntegration.lib.utils import classes
from PlutoIntegration.queries import captures

#label_dir = '/home/kalk/yolov5/runs/detect/exp15/labels'
label_dir = '/home/kalk/yolov5/runs/detect/exp17/labels'

BOLLARD_CODE = 'S40'
BOLLARD_CLASS_INDEX = classes.index(BOLLARD_CODE)

# https://staging.api.plutomap.com/api/v1/annotations?authorType=1
req = {
    "captureId":"c30843f8-a701-4233-b5e7-7f89aab1669c",
    "annotationId":"27cc9b46-03cb-4bab-95be-07df82d00a5e",
    "className":"10",
    "severe":False,
    "isFixed":False,
    "confidence":1,
    "x1":127,
    "x2":197,
    "y1":349,
    "y2":469,
    "bottomMeasure":378.927089006174,
    "leftMeasure":2612.084807389422,
    "longitude":11.832156394179757,
    "latitude":55.78316616156319,
    "xmin":0,
    "crackType":None
}

def relative_to_bbox(x, y, w, h):
    c_x, c_y = x * 1024, y * 1024
    x1, y1 = c_x - (w * 1024) / 2, c_y - (h * 1024) / 2
    x2, y2 = c_x + (w * 1024) / 2, c_y + (h * 1024) / 2
    return x1, y1, x2, y2

def detections_to_labels(capture, detections):
    for label, x, y, w, h, conf in detections:

        x1, y1, x2, y2 = relative_to_bbox(x, y, w, h)

        annotationId = str(uuid.uuid4())
        historyId = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()

        # print(capture.CaptureId, (x1, y1), (x2, y2))
        # print(capture.CaptureId)

        print(" ".join(f"""
        INSERT INTO "Annotations" (
            "AnnotationId",
            "ClassName",
            "Severe",
            "IsFixed",
            "Confidence",
            "X1", "X2", "Y1", "Y2",
            "CaptureId"
        ) VALUES (
            '{annotationId}',
            {BOLLARD_CLASS_INDEX},
            False,
            False,
            {conf},
            {x1}, {x2}, {y1}, {y2},
            '{capture.CaptureId}'
        );
        """.strip().split()).replace('\n', ''))
        print(f"""
        INSERT INTO "AnnotationHistory" (
            "HistoryId",
            "UserId",
            "AnnotationId",
            "CaptureId",
            "AuthorType",
            "OperationType",
            "timestamp",
            "ClassName",
            "X1", "X2", "Y1", "Y2"
        ) VALUES (
            '{historyId}',
            '{capture.UserId}',
            '{annotationId}',
            '{capture.CaptureId}',
            3, -- NOT RELEVANT
            0, -- INSERT Operation
            TO_TIMESTAMP({timestamp}),
            {BOLLARD_CLASS_INDEX},
            {x1}, {x2}, {y1}, {y2}
        );
        """)


with PlutoDB() as c:
    df = captures.getReviewedCaptures(c)
    for det in os.listdir(label_dir):
        imgname = det.split('.')[0]
        capture = df[df.ImgName == imgname].iloc[0]

        with open(os.path.join(label_dir, det)) as f:
            detections = list(map(lambda l: list(map(float, l.strip().split())), f.readlines()))

        detections_to_labels(capture, detections)
