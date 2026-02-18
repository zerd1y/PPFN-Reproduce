import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from albumentations import GaussNoise, Lambda, GaussianBlur, RandomBrightnessContrast, Compose

parser = argparse.ArgumentParser(description="Create corrupt images")
parser.add_argument("--dataset_name", default='HM-TIR', type=str,
                    help="dataset_name")
parser.add_argument("--dataset_dir", default=r'./datasets/', type=str, help="train_dataset_dir")
parser.add_argument("--save_dir", default=r"./datasets/corrupt", type=str, help="Save path of checkpoints")

global opt
opt = parser.parse_args()

def stripe_noise(image, **params):
    if random.random() < 0.5:
        g = np.random.randn(1, image.shape[1]) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image

def nonuniformity_optical(image, **params):
    h, w = image.shape
    noise = np.ones((h, w)).astype("float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4
    if len(image.shape) == 3:
        noise = np.expand_dims(noise, -1)
    if random.random() < 0.5:
        image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
    else:
        image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
    return image

def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(var_limit=(5.0 ** 2,  20.0 ** 2), p=1),
    ], p=p)


def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)


def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1)
    ], p=p)


def create_corrupt():
    # 1. 组合出 test.txt 的位置：./datasets/HM-TIR/test.txt
    dataset_dir = os.path.join(opt.dataset_dir, opt.dataset_name)
    txt_path = os.path.join(dataset_dir, 'test.txt')
    
    # 2. 组合出图片实际存放的目录：./datasets/HM-TIR/test/
    image_folder = os.path.join(dataset_dir, 'test')

    with open(txt_path, 'r') as f:
        img_names = f.read().splitlines()

    tbar = tqdm(img_names)
    for img_name in tbar:
        # 3. 关键修正：确保从 test 文件夹里读取图片
        read_path = os.path.join(image_folder, img_name)
        
        if not os.path.exists(read_path):
            print(f"\n错误：找不到图片 {read_path}")
            continue

        img = Image.open(read_path).convert('L')
        img_list = {'image': np.array(img, dtype=np.uint8)}
        
        # 执行降质处理
        img_list = LC()(**img_list)
        img_list = Blur()(**img_list)
        img_list = Noise()(**img_list)
        
        cor_img = Image.fromarray(img_list['image']).convert('L')

        # 4. 确保保存目录存在
        save_path = os.path.join(opt.save_dir, opt.dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cor_img.save(os.path.join(save_path, img_name))


if __name__ == '__main__':
    print("Creating " + opt.dataset_name + " corrupted images ...")
    create_corrupt()

