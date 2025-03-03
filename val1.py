# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
import utils.model
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils import model
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=detections.device)
    if labels.shape[0]:
        detected = []
        labels = labels.to(detections.device)  # 将 labels 移动到与 detections 相同的设备
        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
        for cls in torch.unique(labels[:, 0]):
            ti = (labels[:, 0] == cls).nonzero(as_tuple=False).view(-1)  # target indices
            pi = (detections[:, 5] == cls).nonzero(as_tuple=False).view(-1)  # prediction indices
            if pi.shape[0]:
                iou, i = box_iou(tbox[ti], detections[pi, :4]).max(1)  # best ious, indices
                detected_set = set()
                for j in (iou > iouv[0]).nonzero(as_tuple=False):
                    d = ti[j]
                    if d not in detected_set:
                        detected_set.add(d)
                        correct[pi[i[j]], :] = (iou[j] > iouv).view(-1)
                        if len(detected_set) == len(ti):
                            break
    return correct



@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for [email protected]:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # padding
        if task == 'speed':
            pass  # no dataloader
        else:
            dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=rect, workers=workers,
                                           prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%12s' * 9) % (
    'Class', 'Images', 'Instances', '[email protected]', '[email protected]:0.95', 'mAP50', 'mAP75', 'AP_small',
    'AP_medium', 'AP_large')

    p, r, f1, mp, mr, map50, map75, maps, mapm, mapl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    losses = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')

    # Run the dataloader
    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)):
        callbacks.run('on_val_batch_start')
        im = im.to(device, non_blocking=True)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        preds = model(im, augment=augment, visualize=False)

        # NMS
        preds = non_max_suppression(preds, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls, max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device)))
                continue

            # Predictions
            if save_txt:
                save_one_txt(pred, save_conf, shape, save_dir / 'labels' / (path.stem + '.txt'))

            if save_json:
                save_one_json(pred, jdict, path, class_map)  # append to COCO-JSON dictionary

            # Evaluate
            correct = process_batch(pred, labels, iouv)
            confusion_matrix.process_batch(pred, labels)
            if nl:
                tcls = labels[:, 0].tolist()  # target class
                correct = process_batch(pred, labels, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], tcls))
            callbacks.run('on_val_image_end', pred, im, targets, paths, shapes, correct)

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            plot_images(im, targets, paths, f, names) if targets.shape[0] else None
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            plot_images(im, output_to_target(preds, paths), paths, f, names) if preds[0] is not None else None
        callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap75 = ap[:, 0], ap[:, 2]
        mp, mr, map50, map75 = p.mean(), r.mean(), ap50.mean(), ap75.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        maps = np.mean(ap[:, 3:6], axis=0)  # APs
        mapm, maps, mapl = maps
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, maps, mapm, mapl))
    for i, c in enumerate(ap_class):
        LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], maps[i], mapm[i], mapl[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in (t0, t1, t2)) + (imgsz, imgsz, batch_size)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape} (%dx%d@%d)' % t)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('val')).parent / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map50, map75 = eval.stats[0], eval.stats[1]
            LOGGER.info(f'pycocotools mAP [email protected]: {map50:.3f}, [email protected]: {map75:.3f}')
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    callbacks.run('on_val_end')
    return (mp, mr, map50, map75, maps, mapm, mapl), ap, ap_class, confusion_matrix


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if opt.task in ('train', 'val', 'test'):  # run normally
        model = load_model(opt.weights).to(device)
        run(model, device, **vars(opt))
    elif opt.task == 'speed':  # speed benchmarks
        for weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
            for d in 'cpu', 'cuda:0':
                model = load_model(weights).to(device)
                opt.device, opt.batch_size = d, 1
                run(model, device, **vars(opt), task='speed')
    elif opt.task == 'study':  # profile a model
        for weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
            model = load_model(weights).to(device)
            run(model, device, **vars(opt), task='study')

def load_model(weights):
    # Load model from weights file
    model = torch.load(weights)['model'].float()
    return model


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)