import os
import json
import cv2
from tqdm import trange

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    label_file = "/home/xc429/datasets/mini_bdd100k/train_100/label.json"
    labels = load_json(label_file)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    for ksize in range(150, 160, 10):
        vis_dir = f"images_lowpass{ksize}_vis"
        os.makedirs(vis_dir, exist_ok=True)
        pred_dir = f"images_lowpass{ksize}_pred"
        os.makedirs(pred_dir, exist_ok=True)
        for idx in trange(100):
            fname = labels[idx]['name'].replace(".jpg", ".png")
            im = cv2.imread(f"/home/xc429/datasets/mini_bdd100k/train_100/images_lowpass{ksize}/{fname}")
            outputs = predictor(im)

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(os.path.join(vis_dir, fname), out.get_image()[:, :, ::-1])

            lines = []
            for i in range(len(outputs["instances"].pred_classes)):
                cname = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)[int(outputs["instances"].pred_classes[i])].capitalize().replace(" ", "")
                bbox = outputs["instances"].pred_boxes.tensor[i].detach().cpu().numpy().tolist()
                line = f"{cname} 0.00 0 0.00 {bbox[0]:0.2f} {bbox[1]:0.2f} {bbox[2]:0.2f} {bbox[3]:0.2f} 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {outputs['instances'].scores[i]:0.2f}"
                lines.append(line)
            with open(os.path.join(pred_dir, fname.replace(".png", ".txt")), "w") as f:
               f.write("\n".join(lines))