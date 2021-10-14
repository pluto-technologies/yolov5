# Pluto YOLOv5

Upstream: ultralytics/yolov5

TODO: Better description
TODO: Experiment should also be the wandb name


### Run

```bash
python detect.py --imgsz 1024 --weights runs/train/exp52/weights/best-int8.tf
lite --iou-thres 0.2 --conf-thres 0.3 --max-det 50 --save-txt --save-conf --source /home/kalk/aab_rout
e
```
