import os
import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class UnetWithGradCAM(object):
    _defaults = {
        "model_path"     : 'logs/best_epoch_weights.pth',
        "num_classes"    : 2,
        "backbone"       : "resnet50",
        "input_shape"    : [512, 512],
        "mix_type"       : 0,
        "cuda"           : False,
        # Grad-CAM layers to hook; update names per your model
        "cam_layers"     : [
            'resnet.layer4.2.conv2',
            'up_concat2.conv2',
            'up_concat2.conv1',
            'resnet.layer4.0.conv2',
            'up_concat2',
            'up_concat2.conv1',
            'up_concat2.up',
            'up_concat2.relu'
        ],
        "cam_output_dir" : 'gradcam_outputs'
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items(): setattr(self, name, value)
        self._features = {}
        self._grads    = {}
        os.makedirs(self.cam_output_dir, exist_ok=True)
        self._build_model()
        show_config(**self._defaults)

    def _hook_layer(self, name):
        layer = dict(self.net.named_modules())[name]
        def forward_hook(m, inp, out):
            if isinstance(out, torch.Tensor): self._features[name] = out.detach().cpu().numpy()
        def backward_hook(m, grad_in, grad_out):
            g = grad_out[0]
            if isinstance(g, torch.Tensor) and g.dim() >= 3: self._grads[name] = g.detach().cpu().numpy()
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def _build_model(self):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.eval(); self.net.to(device)
        module_names = list(dict(self.net.named_modules()).keys())
        print("[Debug] Available module names:")
        for n in module_names: print("  ", n)
        for name in self.cam_layers:
            if name in module_names: self._hook_layer(name)
            else: print(f"Warning: layer '{name}' not found. Skipping.")

    def detect_image(self, image):
        pil_img = cvtColor(image)
        # 保存输入EPS，无边框，300dpi
        eps_in = os.path.join(self.cam_output_dir, 'input.eps')
        pil_img.save(eps_in, format='EPS', dpi=(300,300))

        orig_rgb = np.array(pil_img)
        orig = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        h0, w0 = orig.shape[:2]
        img_resized, nw, nh = resize_image(pil_img, tuple(self.input_shape[::-1]))
        blob = np.transpose(preprocess_input(np.array(img_resized, np.float32)), (2,0,1))[None]
        imgs = torch.from_numpy(blob).to(next(self.net.parameters()).device)
        out = self.net(imgs)[0]; target_score = out.mean()
        self.net.zero_grad(); target_score.backward(retain_graph=True)

        for layer in self.cam_layers:
            if layer not in self._features or layer not in self._grads: continue
            feat = self._features[layer][0]; grad = self._grads[layer][0]
            weights = np.mean(grad, axis=(1,2))
            cam_map = np.zeros(feat.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights): cam_map += w * feat[i]
            cam_map = np.maximum(cam_map,0)
            cam_norm = (cam_map - cam_map.min())/(cam_map.max()-cam_map.min()+1e-8)
            cam_resized = cv2.resize(cam_norm, (w0, h0))
            att_uint8 = (cam_resized*255).astype(np.uint8)
            heat = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
            blend = cv2.addWeighted(orig,0.5,heat,0.5,0)
            # 保存PNG和EPS
            fname = layer.replace('.', '_')
            png_path = os.path.join(self.cam_output_dir, f"{fname}_cam.png")
            eps_path = os.path.join(self.cam_output_dir, f"{fname}_cam.eps")
            cv2.imwrite(png_path, blend)
            # 转RGB并保存EPS
            pil_heat = Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
            pil_heat.save(eps_path, format='EPS', dpi=(300,300))

        # 生成竖直色条
        fig, ax = plt.subplots(figsize=(1,4))
        norm = plt.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=ax)
        cb.outline.set_visible(False)
        plt.tight_layout(pad=0)
        cb_path = os.path.join(self.cam_output_dir, 'colorbar.eps')
        plt.savefig(cb_path, format='eps', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved input and Grad-CAM EPS images in '{self.cam_output_dir}'")

# -------------------- 运行脚本 -------------------- #
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unet with Grad-CAM')
    parser.add_argument('--image', type=str, default='test.jpg', help='Input image path')
    parser.add_argument('--model', type=str, default='logs/best_epoch_weights.pth', help='Model weights')
    parser.add_argument('--output', type=str, default='gradcam_outputs', help='Output directory')
    args = parser.parse_args()
    if not os.path.exists(args.image): print(f"Error: '{args.image}' not found."); exit(1)
    cam = UnetWithGradCAM(model_path=args.model)
    cam.cam_output_dir = args.output; os.makedirs(args.output, exist_ok=True)
    img = Image.open(args.image)
    cam.detect_image(img)
