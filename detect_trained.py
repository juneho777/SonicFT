import os
import sys
import argparse
import json
import cv2
from tqdm import trange

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', type=str, default="lowpass-50", help='lowpass-{kernel_size}/jpeg-{quality_factor}/res-{resolution_scale}')
args = parser.parse_args()


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def get_bdd100k_dicts(
        mode,
        split,
        root="/home/xc429/datasets/bdd100k/bdd100k",
        tracked_classes=["bus", "car", "truck"]):
    if mode.startswith("100k_lowpass"):
        json_file = os.path.join(root, "labels", f"bdd100k_labels_images_{split}.json")
    elif mode.startswith("100k_res"):
        json_file = os.path.join(root, "labels", f"bdd{mode}_labels_images_{split}.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}

        filename = os.path.join(root, "images", mode, split, v["name"])
        if not os.path.isfile(filename):
            filename = filename.replace(".jpg", ".png")

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = 720
        record["width"] = 1280

        annos = v["labels"]
        objs = []
        for anno in annos:
            category = anno["category"]
            if category in tracked_classes:
                anno = anno["box2d"]

                obj = {
                    "bbox": [anno["x1"], anno["y1"], anno["x2"], anno["y2"]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": tracked_classes.index(category),
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for mode in ["100k", "100k_res-0.5", "100k_lowpass-20", "100k_lowpass-30", "100k_lowpass-100", "100k_lowpass-150", "100k_lowpass-200", "100k_lowpass-50"]:
    for split in ["train", "val"]:
        DatasetCatalog.register(f"{mode}_{split}", lambda: get_bdd100k_dicts(mode, split))
        MetadataCatalog.get(f"{mode}_{split}").set(
            thing_classes=["bus", "car", "truck"],
            evaluator_type="coco"
        )


if __name__ == "__main__":
    mode = args.mode  # "100k_lowpass-50"

    label_file = "/home/xc429/datasets/bdd100k/bdd100k/labels/bdd100k_labels_images_val.json"
    labels = load_json(label_file)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model

    weights_dir = f"/home/xc429/repos/detectron2/tools/output_{mode}"
    weights = f"{weights_dir}/model_final.pth"
    if not os.path.isfile(weights):
        flist = [int(x.lstrip("model_").rstrip(".pth")) for x in os.listdir(weights_dir) if x.startswith("model_") and x.endswith(".pth")]
        assert len(flist) > 0, "Unable to find any checkpoint."
        weights = f"model_{max(flist):07d}.pth"
        print(f"Load latest checkpoint {weights}")
        weights = os.path.join(weights_dir, weights)
        assert os.path.isfile(weights)

    cfg.MODEL.WEIGHTS = weights
    cfg.DATASETS.TRAIN = (f"{mode}_train",)
    cfg.DATASETS.TEST = (f"{mode}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    vis_dir = f"{mode}_vis"
    os.makedirs(vis_dir, exist_ok=True)
    pred_dir = f"{mode}_pred"
    os.makedirs(pred_dir, exist_ok=True)
    for idx in trange(100):
        fname = labels[idx]['name'].replace(".jpg", ".png")
        _fname = os.path.join("/home/xc429/datasets/bdd100k/bdd100k", "images", mode, "val", fname)
        im = cv2.imread(_fname)
        outputs = predictor(im)
        # print(_fname, im.shape, outputs["instances"].to("cpu"))

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(f"{mode}_val"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(vis_dir, fname), out.get_image()[:, :, ::-1])

        lines = []
        for i in range(len(outputs["instances"].pred_classes)):
            cname = MetadataCatalog.get(f"{mode}_val").get("thing_classes", None)[int(outputs["instances"].pred_classes[i])].capitalize().replace(" ", "")
            bbox = outputs["instances"].pred_boxes.tensor[i].detach().cpu().numpy().tolist()
            line = f"{cname} 0.00 0 0.00 {bbox[0]:0.2f} {bbox[1]:0.2f} {bbox[2]:0.2f} {bbox[3]:0.2f} 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {outputs['instances'].scores[i]:0.2f}"
            lines.append(line)
        with open(os.path.join(pred_dir, fname.replace(".png", ".txt")), "w") as f:
           f.write("\n".join(lines))