import argparse

import cv2
import numpy as np
import torch
from pathlib import Path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from model import build_model

@torch.no_grad()
def inference(cfg, weight, img, out_file="age_predictions.txt"):
    img_path = img
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    model = build_model(cfg)
    dict_checkpoint = torch.load(weight)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])

    model.eval()
    output = model(img)#.numpy()

    for each in output.keys():
        # print(each, "\t" , output[each][0].numpy())
        if each.startswith("Age"):
            print(img_path, "=>", end=" ")
            print(each, "\t" , output[each][0].numpy())
            age = str(output[each][0].item())

            with open(out_file, "a") as f:
                f.write(img_path + ","+ age + "\n")

class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512

if __name__ == "__main__":
    folder = Path("test_images")
    images = [str(p) for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}]
    print(f"Found {len(images)} images in {folder}")
    cfg = SwinFaceCfg()
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str, default='swinface_project/SwinFace_AgePred.pt')
    args = parser.parse_args()
    for img in images:
        inference(cfg, args.weight, img, out_file="test.txt")

