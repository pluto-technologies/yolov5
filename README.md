# Pluto YOLOv5

Upstream: ultralytics/yolov5

TODO: Better description
TODO: Experiment should also be the wandb name

### Setup

```
pip install -r requirements.txt
pip install git+ssh://git@github.com/pluto-technologies/Pluto-Data.git#egg=PlutoIntegration
```


### Run

```bash
python detect.py --imgsz 1024 --weights runs/train/exp52/weights/best-int8.tflite --iou-thres 0.2 --conf-thres 0.3 --max-det 50 --save-txt --save-conf --source /home/kalk/aab_route
```
