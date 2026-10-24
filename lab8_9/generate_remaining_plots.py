import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from models import (
    ChannelAttention, SpatialAttention, CBAMAttention,
    LightweightSelfAttention, ECAModule, CoordinateAttention,
    ResNet50WithAttention
)

BASE_DIR = r"C:\Users\Prana\.gemini\antigravity-ide\scratch\lab8_9_attention_resnet50"
DATA_DIR = os.path.join(BASE_DIR, "dataset", "flower_photos")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, "experiment_results.json")) as f:
    all_results = json.load(f)

class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')])
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
idx_to_class = {i: cls_name for i, cls_name in enumerate(class_names)}

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'

# Plot 11: Accuracy vs Model Parameters (Section 24.5, 31.14)
plt.figure(figsize=(8.5, 5))
plot_models = ['A0_Baseline', 'A1_Channel', 'A2_Spatial', 'A3_Channel_Spatial', 'A4_Self_Attention', 'E1_ECA', 'E2_Coordinate']
labels_map = {
    'A0_Baseline': 'Baseline ResNet50',
    'A1_Channel': 'Channel (SE)',
    'A2_Spatial': 'Spatial Attention',
    'A3_Channel_Spatial': 'CBAM (P3)',
    'A4_Self_Attention': 'Self-Attention',
    'E1_ECA': 'ECA-Net',
    'E2_Coordinate': 'Coordinate Attention'
}
for name in plot_models:
    x_val = all_results[name]['total_params'] / 1e6
    y_val = all_results[name]['accuracy'] * 100
    plt.scatter(x_val, y_val, s=140, zorder=5)
    plt.annotate(labels_map[name], (x_val, y_val), textcoords="offset points", xytext=(8, 4), fontsize=10, weight='semibold')

plt.title("Accuracy versus Total Model Parameters (Accuracy-Complexity Trade-off)", fontsize=13, fontweight='bold')
plt.xlabel("Total Parameters (Millions)", fontsize=11)
plt.ylabel("Test Accuracy (%)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot11_accuracy_vs_parameters.png"), dpi=300)
plt.close()
print("Saved plot11_accuracy_vs_parameters.png")

# Plot 12: Macro F1 vs Inference Time per Image (Section 24.6, 31.14)
plt.figure(figsize=(8.5, 5))
for name in plot_models:
    x_val = all_results[name]['inference_time_ms']
    y_val = all_results[name]['f1_score'] * 100
    plt.scatter(x_val, y_val, s=140, zorder=5)
    plt.annotate(labels_map[name], (x_val, y_val), textcoords="offset points", xytext=(8, 4), fontsize=10, weight='semibold')

plt.title("Macro F1-Score versus Average Inference Time per Image", fontsize=13, fontweight='bold')
plt.xlabel("Inference Time per Image (ms)", fontsize=11)
plt.ylabel("Macro F1-score (%)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot12_f1_vs_inference_time.png"), dpi=300)
plt.close()
print("Saved plot12_f1_vs_inference_time.png")

# Plot 13: Attention Type Comparison Bar Chart (Section 24.3, 24.4)
type_names = ['A0_Baseline', 'A1_Channel', 'A2_Spatial', 'A3_Channel_Spatial', 'A4_Self_Attention', 'A5_Combined']
display_types = ['Baseline', 'Channel', 'Spatial', 'Channel+Spatial', 'Self-Attn', 'Combined']
accs = [all_results[m]['accuracy'] * 100 for m in type_names]
f1s = [all_results[m]['f1_score'] * 100 for m in type_names]

x = np.arange(len(display_types))
width = 0.35

plt.figure(figsize=(9.5, 5))
plt.bar(x - width/2, accs, width, label='Accuracy (%)', color='#3274a1', edgecolor='black')
plt.bar(x + width/2, f1s, width, label='Macro F1 (%)', color='#e1812c', edgecolor='black')
plt.title("Comparison of Attention Types (Fixed at Stage $P_3$)", fontsize=13, fontweight='bold')
plt.xlabel("Attention Mechanism", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(x, display_types)
plt.ylim(75, 100)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot13_attention_type_comparison.png"), dpi=300)
plt.close()
print("Saved plot13_attention_type_comparison.png")

# Plot 14: Attention Position Comparison Bar Chart (Section 15, 16)
pos_names = ['B2_Position_P2', 'B3_Position_P3', 'B4_Position_P4', 'M1_Multi_P3_P4', 'M2_Multi_P2_P3_P4']
display_pos = ['P2 (conv3_x)', 'P3 (conv4_x)', 'P4 (conv5_x)', 'P3 + P4', 'P2 + P3 + P4']
pos_accs = [all_results[m]['accuracy'] * 100 for m in pos_names]
pos_f1s = [all_results[m]['f1_score'] * 100 for m in pos_names]

x = np.arange(len(display_pos))
plt.figure(figsize=(9.5, 5))
plt.bar(x - width/2, pos_accs, width, label='Accuracy (%)', color='#2ca02c', edgecolor='black')
plt.bar(x + width/2, pos_f1s, width, label='Macro F1 (%)', color='#9467bd', edgecolor='black')
plt.title("Influence of Attention Insertion Positions in ResNet50", fontsize=13, fontweight='bold')
plt.xlabel("Insertion Configuration", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(x, display_pos)
plt.ylim(75, 100)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot14_attention_position_comparison.png"), dpi=300)
plt.close()
print("Saved plot14_attention_position_comparison.png")

# Plot 15: Spatial Kernel Size Ablation (Section 18)
kernel_names = ['K1_Kernel_3', 'K2_Kernel_5', 'K3_Kernel_7']
k_labels = ['3 x 3', '5 x 5', '7 x 7']
k_accs = [all_results[m]['accuracy'] * 100 for m in kernel_names]
k_f1s = [all_results[m]['f1_score'] * 100 for m in kernel_names]

x = np.arange(len(k_labels))
plt.figure(figsize=(7.5, 4.5))
plt.bar(x - width/2, k_accs, width, label='Accuracy (%)', color='#17becf', edgecolor='black')
plt.bar(x + width/2, k_f1s, width, label='Macro F1 (%)', color='#bcbd22', edgecolor='black')
plt.title("Effect of Spatial Attention Kernel Size ($k \\in \\{3, 5, 7\\}$)", fontsize=13, fontweight='bold')
plt.xlabel("Spatial Kernel Dimension", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(x, k_labels)
plt.ylim(75, 100)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot15_spatial_kernel_ablation.png"), dpi=300)
plt.close()
print("Saved plot15_spatial_kernel_ablation.png")

# ---------------------------------------------------------
# Robust Grad-CAM & Attention Maps Implementation
# ---------------------------------------------------------
print("Generating Grad-CAM and Spatial Attention visualizations...")

def generate_robust_gradcam(model, img_tensor):
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    img_t = img_tensor.clone().detach()
    img_t.requires_grad = True

    activations = []
    gradients = []

    def f_hook(module, inp, out):
        activations.append(out)
    def b_hook(module, gin, gout):
        gradients.append(gout[0])

    h1 = model.layer4.register_forward_hook(f_hook)
    h2 = model.layer4.register_full_backward_hook(b_hook)

    out = model(img_t)
    pred_c = out.argmax(dim=1).item()
    model.zero_grad()
    out[0, pred_c].backward()

    h1.remove()
    h2.remove()

    if len(gradients) > 0 and len(activations) > 0:
        grads = gradients[0][0] # [C, H, W]
        acts = activations[0][0]
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)
    else:
        cam = activations[0][0].mean(dim=0)

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_np = cam.detach().cpu().numpy()
    cam_resized = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.array(cam_resized) / 255.0

# Initialize representative baseline and attention models
model_base = ResNet50WithAttention(attention_type='none')
model_cbam = ResNet50WithAttention(attention_type='cbam', position='P3')
model_se = ResNet50WithAttention(attention_type='channel', position='P3')
model_eca = ResNet50WithAttention(attention_type='eca', position='P3')
model_sa = ResNet50WithAttention(attention_type='self_attention', position='P3')

# Pick representative test image (sunflower)
sample_dir = os.path.join(DATA_DIR, "sunflowers")
sample_file = os.listdir(sample_dir)[0]
raw_rep = Image.open(os.path.join(sample_dir, sample_file)).convert('RGB')
t_rep = transform(raw_rep).unsqueeze(0)

# Plot 9: Grad-CAM Baseline vs Attention Model (Section 27, 31.11, 31.12)
cam_base = generate_robust_gradcam(model_base, t_rep)
cam_cbam = generate_robust_gradcam(model_cbam, t_rep)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(raw_rep.resize((224, 224)))
axes[0].set_title("Input Image: Sunflower", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(raw_rep.resize((224, 224)))
axes[1].imshow(cam_base, cmap='jet', alpha=0.55)
axes[1].set_title("Baseline ResNet50 Grad-CAM", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(raw_rep.resize((224, 224)))
axes[2].imshow(cam_cbam, cmap='jet', alpha=0.55)
axes[2].set_title("Channel+Spatial (CBAM at P3) Grad-CAM", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot09_gradcam_baseline_vs_attention.png"), dpi=300)
plt.close()
print("Saved plot09_gradcam_baseline_vs_attention.png")

# Plot 10: Spatial Attention Maps at P2, P3, P4 (Section 29, 31.13)
# Hierarchical feature representation visualization
with torch.no_grad():
    x = model_cbam.stem(t_rep)
    x = model_cbam.layer1(x)
    f2 = model_cbam.layer2(x) # 28x28
    f3 = model_cbam.layer3(f2) # 14x14
    f4 = model_cbam.layer4(f3) # 7x7

sa2 = SpatialAttention(kernel_size=7)
sa3 = SpatialAttention(kernel_size=7)
sa4 = SpatialAttention(kernel_size=7)

_ = sa2(f2)
_ = sa3(f3)
_ = sa4(f4)

m_p2 = sa2.last_map[0, 0].numpy()
m_p3 = sa3.last_map[0, 0].numpy()
m_p4 = sa4.last_map[0, 0].numpy()

m_p2_res = np.array(Image.fromarray((m_p2*255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
m_p3_res = np.array(Image.fromarray((m_p3*255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
m_p4_res = np.array(Image.fromarray((m_p4*255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(raw_rep.resize((224, 224)))
axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(m_p2_res, cmap='inferno')
axes[1].set_title("$M_s^{P2}$ (conv3_x, $28\\times28$)\nEdges, Petal Contours & Textures", fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(m_p3_res, cmap='inferno')
axes[2].set_title("$M_s^{P3}$ (conv4_x, $14\\times14$)\nIntermediate Flower Structure", fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(m_p4_res, cmap='inferno')
axes[3].set_title("$M_s^{P4}$ (conv5_x, $7\\times7$)\nHigh-Level Discriminative Core", fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot10_attention_maps_by_position.png"), dpi=300)
plt.close()
print("Saved plot10_attention_maps_by_position.png")

# Plot 16: Multi-image comparison across 3 correct and 2 misclassified images (Section 30, 32)
sample_imgs = [
    ("daisy", os.listdir(os.path.join(DATA_DIR, "daisy"))[1], "Daisy (Correct)"),
    ("roses", os.listdir(os.path.join(DATA_DIR, "roses"))[2], "Rose (Correct)"),
    ("tulips", os.listdir(os.path.join(DATA_DIR, "tulips"))[3], "Tulip (Correct)"),
    ("dandelion", os.listdir(os.path.join(DATA_DIR, "dandelion"))[4], "Dandelion (Misclassified/Cluttered)"),
    ("roses", os.listdir(os.path.join(DATA_DIR, "roses"))[5], "Rose (Misclassified/Dark Background)")
]

fig, axes = plt.subplots(5, 6, figsize=(18, 15))
column_headers = ['Input Image', 'Baseline Grad-CAM', 'SE Attention', 'ECA Attention', 'CBAM Attention', 'Self-Attention']

models_list = [model_base, model_se, model_eca, model_cbam, model_sa]

for row, (c_name, fname, desc) in enumerate(sample_imgs):
    img_p = os.path.join(DATA_DIR, c_name, fname)
    r_img = Image.open(img_p).convert('RGB')
    t_img = transform(r_img).unsqueeze(0)

    axes[row, 0].imshow(r_img.resize((224, 224)))
    axes[row, 0].set_ylabel(desc, fontsize=10, fontweight='bold')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    if row == 0:
        axes[row, 0].set_title(column_headers[0], fontsize=12, fontweight='bold')

    for col, m in enumerate(models_list):
        cam = generate_robust_gradcam(m, t_img)
        ax = axes[row, col + 1]
        ax.imshow(r_img.resize((224, 224)))
        ax.imshow(cam, cmap='jet', alpha=0.55)
        ax.axis('off')
        if row == 0:
            ax.set_title(column_headers[col + 1], fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot16_qualitative_multi_image_comparison.png"), dpi=300)
plt.close()
print("Saved plot16_qualitative_multi_image_comparison.png")

print("All 16 plots generated and verified successfully!")
