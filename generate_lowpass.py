import os
import numpy as np
import json
from tqdm import tqdm, trange
import argparse
import cv2
import torch
import torchvision
from PIL import Image
from imageio import imread, imwrite


parser = argparse.ArgumentParser(description='generate lowpass')
parser.add_argument('--mode', type=str, default="lowpass-50", help='lowpass-{kernel_size}/jpeg-{quality_factor}/res-{resolution_scale}')
args = parser.parse_args()


def get_coord(n):
    assert n > 0, "invalid lowpass kernel size"
    xs, ys = [], []
    for i in range(n):
        for j in range(i+1):
            xs.append(j)
            ys.append(i-j)
    return torch.tensor(xs), torch.tensor(ys)


def complex_inv(x):
    y = x.clone()
    y.imag = -y.imag
    return y


def compress(img, xs, ys):
    ftimg = torch.fft.fft2(img)
    cpr = ftimg[:, xs, ys]
    cpr2 = ftimg[:, -1-xs, ys+1]
    cpr = torch.stack([cpr, cpr2])
    cpr = torch.stack([cpr.real, cpr.imag])
    return cpr


def extract(cpr, xs, ys, h=720, w=1280):
    c = cpr.shape[2]
    cpr = torch.complex(cpr[0], cpr[1])
    recon = torch.zeros((c, h ,w), dtype=torch.complex64).to(cpr.device)
    recon[:, xs+1, -ys-1] = complex_inv(cpr[1])
    recon[:, -1-xs, ys+1] = cpr[1]
    recon[:, -xs, -ys] = complex_inv(cpr[0])
    recon[:, xs, ys] = cpr[0]
    img = torch.fft.ifft2(recon).real
    img = torch.clamp(img, 0, 1)
    return img


def compress_img(img, xs, ys):
    _, h, w = img.shape
    cpr = compress(img.cuda(), xs, ys)
    return extract(cpr, xs, ys, h, w).detach().cpu()


if __name__ == "__main__":
    root = "/home/xc429/datasets/bdd100k/bdd100k/images"
    label_root = "/home/xc429/datasets/bdd100k/bdd100k/labels"
    print(f"Converting to {args.mode}")
    if args.mode.startswith("lowpass"):
        ksize = int(args.mode.split("-")[1])
        xs, ys = get_coord(ksize)
    elif args.mode.startswith("res"):
        scale = float(args.mode.split("-")[1])
        scale_f = lambda x: int(round(x * scale))

    for split in ["train", "val"]:
        print(f"Converting split {split}.")
        src = os.path.join(root, "100k", split)
        dst = os.path.join(root, f"100k_{args.mode}", split)
        os.makedirs(dst, exist_ok=True)
        for i, fname in tqdm(enumerate(os.listdir(src))):
            target = os.path.join(dst, fname.replace(".jpg", ".png"))
            if not os.path.isfile(target):# or cv2.imread(target).shape[0] != 720:
                img = torchvision.transforms.ToTensor()(Image.open(os.path.join(src, fname)))
                if args.mode.startswith("lowpass"):
                    cimg = compress_img(img, xs, ys)
                    cimg = (cimg.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                elif args.mode.startswith("res"):
                    cimg = img.numpy().transpose(1, 2, 0)
                    h, w, _ = cimg.shape
                    h, w = scale_f(h), scale_f(w)
                    cimg = cv2.resize(cimg, (w, h))
                    cimg = (cimg * 255).astype(np.uint8)
                imwrite(target, cimg)
        if args.mode.startswith("res"):
            with open(os.path.join(label_root, f"bdd100k_labels_images_{split}.json")) as f:
                labels = json.load(f)
            for i in trange(len(labels)):
                for j in range(len(labels[i]["labels"])):
                    if "box2d" in labels[i]["labels"][j]:
                        for k in labels[i]["labels"][j]["box2d"].keys():
                            labels[i]["labels"][j]["box2d"][k] = labels[i]["labels"][j]["box2d"][k] * scale
            with open(os.path.join(label_root, f"bdd100k_{args.mode}_labels_images_{split}.json"), "w") as f:
                json.dump(labels, f, indent=4)