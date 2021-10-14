import os
import numpy as np
from tqdm import tqdm
import torch
from utils.torch_utils import select_device
from val import process_batch
from utils.general import xywh2xyxy
from utils.metrics import ap_per_class, ConfusionMatrix

# detections_dir = '/home/kalk/yolov5/runs/detect/exp53/labels/'
detections_dir = '/home/kalk/yolov5/runs/detect/exp72/labels/'
labels_dir = '/home/kalk/data/pluto/labels'
#images_file = '/home/kalk/data/pluto/val.txt'
images_file = '/home/kalk/data/pluto/val_sample.txt'
#OUTDIR = 'pluto4ai'
OUTDIR = 'tflite-sample'

device = select_device('', batch_size=32)


def read_yolo_lines(fh):
    return [list(map(float, line.strip().split())) for line in fh]

def get_detections(img_name):
    detection_file = os.path.join(detections_dir, os.path.basename(img_name.replace('png', 'txt')))
    if not os.path.isfile(detection_file):
        return None

    with open(os.path.join(detections_dir, detection_file)) as f:
        detections = read_yolo_lines(f)

    return translate_class_index(detections)

def translate_class_index(detections):
    pluto2ai4eu = { i: v for i, v in enumerate(names)}
      ##pluto2ai4eu = {
      ##    1: 1, # D40
      ##    6: 5, # D10
      ##    11: 2, # M10
      ##    12: 3 # M20
      ##}
    for i in range(len(detections)):
        cls = detections[i][0]
        if cls in pluto2ai4eu:
            detections[i][0] = pluto2ai4eu[cls]
        else:
            raise ValueError(f"{cls} not found")
    dets = np.array(detections)
    xyxy = xywh2xyxy(dets[:, 1:5])
    l = dets[:, 0:1]
    c = dets[:, 5:]
    return np.concatenate((xyxy, c, l), axis=1)

def get_labels(label_img):
    bbox_labels = []
    label_file = label_img.replace('png', 'txt').replace('images', 'labels')

    with open(label_file) as f :
        bbox_labels += read_yolo_lines(f) or [[0 for _ in range(5)]]

    bbox_labels = np.array(bbox_labels)
    return np.concatenate((bbox_labels[:, 0:1], xywh2xyxy(bbox_labels[:, 1:5])), axis=1)


def compare_img(img_name, iouv):
    labels = get_labels(img_name)
    dets = get_detections(img_name)
    correct = process_batch(torch.from_numpy(dets).to(device), torch.from_numpy(labels).to(device), iouv)
    return correct


#names = ['Rutting', 'D40', 'M10', 'M20', 'EdgeDeterioration', 'D10']
names = [
  "A20",
  "D40",
  "D50",
  "A10",
  "P20",
  "D20",
  "D10",
  "R30",
  "S10",
  "D30",
  "R10",
  "M10",
  "M20",
  "S20",
  "R20",
  "S30",
  "S40",
]

if __name__ == '__main__':
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)

    iouv = torch.linspace(0.01, 0.5, 10).to(device)  # iou vector for mAP@0.5:0.95
    stats = []
    confusion_matrix = ConfusionMatrix(nc=len(names))
    names = {k: v for k, v in enumerate(names)}
    with open(images_file) as f:
        image_names = [line.strip() for line in f]

     #  bbox_labels = get_labels(labels_dir, labels)
     #  xyxy = xywh2xyxy(np.array(bbox_labels)[:, 1:])
     #  l = np.array(bbox_labels)[:, 0:1]

    # labels = np.concatenate((l, xyxy), axis=1)
    seen = 0
    for img_name in tqdm(image_names):
        seen += 1
        labels = torch.from_numpy(get_labels(img_name)).to(device)
        dets = get_detections(img_name)
        if dets is None: continue
        dets = torch.from_numpy(dets).to(device)
        correct = process_batch(dets, labels, iouv)
        stats.append((correct.cpu(), dets[:, 4].cpu(), dets[:, 5].cpu(), labels[:, 0].clone().cpu()))  # (correct, conf, pcls, tcls)

        confusion_matrix.process_batch(dets, labels)

    confusion_matrix.plot(save_dir=OUTDIR, names=list(names.values()))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=OUTDIR, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(names))  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.1:.5')
    print(s)
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
