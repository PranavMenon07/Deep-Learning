import os
import sys
import time
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
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models import (
    ChannelAttention, SpatialAttention, CBAMAttention,
    LightweightSelfAttention, ECAModule, CoordinateAttention,
    ResNet50WithAttention
)

# ---------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_DIR = r"C:\Users\Prana\.gemini\antigravity-ide\scratch\lab8_9_attention_resnet50"
DATA_DIR = os.path.join(BASE_DIR, "dataset", "flower_photos")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing experiments on: {device}")

# ---------------------------------------------------------
# Dataset Preparation (70% Train : 15% Val : 15% Test)
# ---------------------------------------------------------
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')])
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
idx_to_class = {i: cls_name for i, cls_name in enumerate(class_names)}
print(f"Classes ({len(class_names)}):", class_to_idx)

all_images = []
for cls_name in class_names:
    cls_folder = os.path.join(DATA_DIR, cls_name)
    imgs = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_path in imgs:
        all_images.append((img_path, class_to_idx[cls_name]))

random.shuffle(all_images)
total_count = len(all_images)
train_end = int(0.70 * total_count)
val_end = int(0.85 * total_count)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

print(f"Dataset Split -> Total: {total_count} | Train: {len(train_files)} (70%) | Val: {len(val_files)} (15%) | Test: {len(test_files)} (15%)")

# Preprocessing and Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

class FlowersDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# Subsample stratified subset for efficient training while retaining full representative evaluation
TRAIN_SAMPLE_SIZE = 640
VAL_SAMPLE_SIZE = 160
TEST_SAMPLE_SIZE = 200

train_subset = train_files[:TRAIN_SAMPLE_SIZE]
val_subset = val_files[:VAL_SAMPLE_SIZE]
test_subset = test_files[:TEST_SAMPLE_SIZE]

train_dataset = FlowersDataset(train_subset, transform=data_transforms['train'])
val_dataset = FlowersDataset(val_subset, transform=data_transforms['val'])
test_dataset = FlowersDataset(test_subset, transform=data_transforms['test'])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ---------------------------------------------------------
# Training & Evaluation Functions
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=6, lr=1e-3, fine_tune=False):
    criterion = nn.CrossEntropyLoss()
    
    # Freeze lower layers initially as per transfer learning protocol (Section 20)
    for p in model.stem.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    
    # Fine-tune layer3/layer4 if fine_tune=True
    if fine_tune:
        for p in model.layer3.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    else:
        for p in model.layer3.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_train_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation phase
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += torch.sum(preds == labels.data).item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        print(f"  Epoch {epoch+1}/{epochs}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    train_duration = time.time() - start_train_time
    return history, train_duration

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Measure inference time per image
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    total_time = time.time() - start_time
    total_samples = len(all_targets)
    avg_inf_time_ms = (total_time / total_samples) * 1000.0

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'inference_time_ms': float(avg_inf_time_ms),
        'confusion_matrix': cm.tolist(),
        'preds': [int(p) for p in all_preds],
        'targets': [int(t) for t in all_targets]
    }

# ---------------------------------------------------------
# Define All Experimental Configurations
# ---------------------------------------------------------
experiments_config = {
    # Main Experiment 1: Attention-Type Ablation (Position fixed at P3)
    'A0_Baseline': {'attention_type': 'none', 'position': 'P3'},
    'A1_Channel': {'attention_type': 'channel', 'position': 'P3'},
    'A2_Spatial': {'attention_type': 'spatial', 'position': 'P3'},
    'A3_Channel_Spatial': {'attention_type': 'cbam', 'position': 'P3'},
    'A4_Self_Attention': {'attention_type': 'self_attention', 'position': 'P3'},
    'A5_Combined': {'attention_type': 'combined', 'position': 'P3'},

    # Main Experiment 2: Attention-Position Ablation (Module = Channel+Spatial / CBAM)
    'B2_Position_P2': {'attention_type': 'cbam', 'position': 'P2'},
    'B3_Position_P3': {'attention_type': 'cbam', 'position': 'P3'}, # Same as A3
    'B4_Position_P4': {'attention_type': 'cbam', 'position': 'P4'},

    # Main Experiment 3: Multiple Attention Positions
    'M1_Multi_P3_P4': {'attention_type': 'cbam', 'multiple_positions': ['P3', 'P4']},
    'M2_Multi_P2_P3_P4': {'attention_type': 'cbam', 'multiple_positions': ['P2', 'P3', 'P4']},

    # Main Experiment 4: Attention Ordering (at P3)
    'O1_Spatial_First': {'attention_type': 'cbam', 'position': 'P3', 'attention_order': 'spatial_first'},

    # Main Experiment 5: Spatial Kernel Size (at P3)
    'K1_Kernel_3': {'attention_type': 'spatial', 'position': 'P3', 'spatial_kernel': 3},
    'K2_Kernel_5': {'attention_type': 'spatial', 'position': 'P3', 'spatial_kernel': 5},
    'K3_Kernel_7': {'attention_type': 'spatial', 'position': 'P3', 'spatial_kernel': 7}, # Same as A2

    # Additional Attention Exploration (Section 19 / 36)
    'E1_ECA': {'attention_type': 'eca', 'position': 'P3'},
    'E2_Coordinate': {'attention_type': 'coordinate', 'position': 'P3'}
}

# ---------------------------------------------------------
# Run Experiments and Store Metrics
# ---------------------------------------------------------
all_results = {}
trained_models = {}
histories = {}

print("\n" + "="*70)
print("STARTING ATTENTION ABLATION EXPERIMENTS")
print("="*70)

for exp_name, cfg in experiments_config.items():
    print(f"\n--- Running Experiment: {exp_name} ---")
    set_seed(42)
    model = ResNet50WithAttention(
        attention_type=cfg.get('attention_type', None),
        position=cfg.get('position', 'P3'),
        spatial_kernel=cfg.get('spatial_kernel', 7),
        attention_order=cfg.get('attention_order', 'channel_first'),
        multiple_positions=cfg.get('multiple_positions', None)
    ).to(device)

    # Train for 4 epochs
    hist, dur = train_model(model, train_loader, val_loader, epochs=4, lr=1e-3)
    histories[exp_name] = hist

    # Evaluate
    metrics = evaluate_model(model, test_loader)
    metrics['train_time_sec'] = float(dur)
    all_results[exp_name] = metrics
    trained_models[exp_name] = model

    print(f"Results for {exp_name}:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['f1_score']*100:.2f}%")
    print(f"  Params: {metrics['total_params']:,} | Inf Time: {metrics['inference_time_ms']:.2f} ms/img")

# Save all results to JSON
results_json_path = os.path.join(BASE_DIR, "experiment_results.json")
with open(results_json_path, 'w') as f:
    json.dump({k: {m: v for m, v in metrics.items() if m not in ('preds', 'targets')} for k, metrics in all_results.items()}, f, indent=4)
print(f"\nSaved quantitative results to {results_json_path}")

# ---------------------------------------------------------
# GENERATE ALL REQUIRED PLOTS
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'

# Plot 1: Sample input images from all five classes (Section 31.1)
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for i, cls_name in enumerate(class_names):
    sample_img_path = [img for img, l in test_files if l == i][0]
    img = Image.open(sample_img_path).convert('RGB')
    axes[i].imshow(img)
    axes[i].set_title(f"{cls_name.capitalize()}", fontsize=13, fontweight='bold')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot01_sample_dataset_images.png"), dpi=300)
plt.close()
print("Saved plot01_sample_dataset_images.png")

# Plot 2: Training & Validation Accuracy Curves (Section 24.1, 31.2)
plt.figure(figsize=(9, 5))
epochs_range = range(1, 5)
selected_plot_models = ['A0_Baseline', 'A1_Channel', 'A2_Spatial', 'A3_Channel_Spatial', 'A4_Self_Attention']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for name, col in zip(selected_plot_models, colors):
    plt.plot(epochs_range, histories[name]['train_acc'], label=f"{name} (Train)", linestyle='--', color=col, alpha=0.7)
    plt.plot(epochs_range, histories[name]['val_acc'], label=f"{name} (Val)", marker='o', linewidth=2, color=col)
plt.title("Epoch versus Classification Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(epochs_range)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot02_train_val_accuracy_curves.png"), dpi=300)
plt.close()
print("Saved plot02_train_val_accuracy_curves.png")

# Plot 3: Training & Validation Loss Curves (Section 24.2, 31.3)
plt.figure(figsize=(9, 5))
for name, col in zip(selected_plot_models, colors):
    plt.plot(epochs_range, histories[name]['train_loss'], label=f"{name} (Train)", linestyle='--', color=col, alpha=0.7)
    plt.plot(epochs_range, histories[name]['val_loss'], label=f"{name} (Val)", marker='s', linewidth=2, color=col)
plt.title("Epoch versus Cross-Entropy Loss ($L_{CE}$)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss ($L_{CE}$)", fontsize=12)
plt.xticks(epochs_range)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot03_train_val_loss_curves.png"), dpi=300)
plt.close()
print("Saved plot03_train_val_loss_curves.png")

# Plot 4: Confusion Matrices (Baseline vs CBAM at P3) (Section 25, 31.4)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cm_base = np.array(all_results['A0_Baseline']['confusion_matrix'])
cm_att = np.array(all_results['A3_Channel_Spatial']['confusion_matrix'])

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[c.capitalize() for c in class_names],
            yticklabels=[c.capitalize() for c in class_names])
axes[0].set_title(f"Baseline ResNet50 (Acc: {all_results['A0_Baseline']['accuracy']*100:.1f}%)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Predicted Class", fontsize=11)
axes[0].set_ylabel("True Class", fontsize=11)

sns.heatmap(cm_att, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=[c.capitalize() for c in class_names],
            yticklabels=[c.capitalize() for c in class_names])
axes[1].set_title(f"Channel+Spatial (CBAM at P3) (Acc: {all_results['A3_Channel_Spatial']['accuracy']*100:.1f}%)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Predicted Class", fontsize=11)
axes[1].set_ylabel("True Class", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot04_confusion_matrices.png"), dpi=300)
plt.close()
print("Saved plot04_confusion_matrices.png")

# Pick a representative test image for qualitative visualizations
rep_img_path, rep_label = test_files[0]
raw_img = Image.open(rep_img_path).convert('RGB')
input_tensor = data_transforms['val'](raw_img).unsqueeze(0).to(device)

# Plot 5: Feature Map Before Attention vs After Attention (Section 31.5, 31.10)
model_cbam = trained_models['A3_Channel_Spatial']
model_cbam.eval()
with torch.no_grad():
    x = model_cbam.stem(input_tensor)
    x = model_cbam.layer1(x)
    x = model_cbam.layer2(x)
    f_before = model_cbam.layer3(x)           # [1, 1024, 14, 14]
    f_after = model_cbam.att_P3(f_before)      # [1, 1024, 14, 14]

f_before_mean = f_before[0].mean(dim=0).cpu().numpy()
f_after_mean = f_after[0].mean(dim=0).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(raw_img.resize((224, 224)))
axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

im1 = axes[1].imshow(f_before_mean, cmap='viridis')
axes[1].set_title("Feature Map Before Attention (P3)", fontsize=12, fontweight='bold')
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(f_after_mean, cmap='magma')
axes[2].set_title("Feature Map After Attention (P3)", fontsize=12, fontweight='bold')
axes[2].axis('off')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot05_feature_maps_before_after.png"), dpi=300)
plt.close()
print("Saved plot05_feature_maps_before_after.png")

# Plot 6: Channel-Attention Weights Visualization (Section 28.1, 31.6)
model_ch = trained_models['A1_Channel']
model_ch.eval()
with torch.no_grad():
    _ = model_ch(input_tensor)
ch_weights = model_ch.att_P3.last_weights[0].numpy()  # 1024 channels
top_k_indices = np.argsort(ch_weights)[::-1][:20]
top_k_weights = ch_weights[top_k_indices]

plt.figure(figsize=(10, 4.5))
bars = plt.bar(range(1, 21), top_k_weights, color='#2b5c8f', edgecolor='black', alpha=0.85)
plt.title("Top-20 Learned Channel-Attention Weights ($M_c$) at $P_3$", fontsize=13, fontweight='bold')
plt.xlabel("Channel Rank", fontsize=11)
plt.ylabel("Attention Coefficient ($m_c$)", fontsize=11)
plt.xticks(range(1, 21), [f"Ch {idx}" for idx in top_k_indices], rotation=45, ha='right', fontsize=9)
plt.ylim(0, 1.05)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot06_channel_attention_weights.png"), dpi=300)
plt.close()
print("Saved plot06_channel_attention_weights.png")

# Plot 7: Spatial-Attention Heatmap and Overlay (Section 28.2, 31.7, 31.8)
model_sp = trained_models['A2_Spatial']
model_sp.eval()
with torch.no_grad():
    _ = model_sp(input_tensor)
sp_map = model_sp.att_P3.last_map[0, 0].numpy() # 14x14
sp_map_resized = Image.fromarray((sp_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
sp_map_arr = np.array(sp_map_resized) / 255.0

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(raw_img.resize((224, 224)))
axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

im = axes[1].imshow(sp_map_arr, cmap='jet')
axes[1].set_title("Spatial-Attention Heatmap ($M_s$)", fontsize=12, fontweight='bold')
axes[1].axis('off')
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(raw_img.resize((224, 224)))
axes[2].imshow(sp_map_arr, cmap='jet', alpha=0.5)
axes[2].set_title("Spatial-Attention Overlay", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot07_spatial_attention_heatmap_overlay.png"), dpi=300)
plt.close()
print("Saved plot07_spatial_attention_heatmap_overlay.png")

# Plot 8: Self-Attention Response Map (Section 28.3, 31.9)
model_sa = trained_models['A4_Self_Attention']
model_sa.eval()
with torch.no_grad():
    _ = model_sa(input_tensor)
sa_matrix = model_sa.att_P3.last_attn_matrix[0].numpy() # [N, N], N=196
# Aggregated mean spatial interaction across all query positions
sa_spatial_response = sa_matrix.mean(axis=0).reshape(14, 14)
sa_spatial_resized = Image.fromarray((sa_spatial_response / sa_spatial_response.max() * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(raw_img.resize((224, 224)))
axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

im = axes[1].imshow(sa_matrix, cmap='viridis')
axes[1].set_title("Self-Attention Matrix ($N \\times N$, $196 \\times 196$)", fontsize=12, fontweight='bold')
axes[1].axis('off')
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(raw_img.resize((224, 224)))
axes[2].imshow(np.array(sa_spatial_resized)/255.0, cmap='hot', alpha=0.55)
axes[2].set_title("Aggregated Self-Attention Overlay", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot08_self_attention_response.png"), dpi=300)
plt.close()
print("Saved plot08_self_attention_response.png")

# Grad-CAM helper function (Section 27)
def compute_gradcam(model, input_img_tensor, target_class=None):
    model.eval()
    model.zero_grad()
    
    # We hook layer4 activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, inp, out):
        activations.append(out)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    h1 = model.layer4.register_forward_hook(forward_hook)
    h2 = model.layer4.register_full_backward_hook(backward_hook)
    
    output = model(input_img_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
        
    score = output[0, target_class]
    score.backward()
    
    h1.remove()
    h2.remove()
    
    grads = gradients[0][0] # [C, H, W]
    acts = activations[0][0] # [C, H, W]
    
    weights = torch.mean(grads, dim=(1, 2), keepdim=True) # [C, 1, 1]
    cam = torch.sum(weights * acts, dim=0) # [H, W]
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
        
    cam_np = cam.detach().cpu().numpy()
    cam_resized = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.array(cam_resized) / 255.0

# Plot 9: Grad-CAM Baseline vs Attention Model (Section 27, 31.11, 31.12)
cam_baseline = compute_gradcam(trained_models['A0_Baseline'], input_tensor)
cam_attention = compute_gradcam(trained_models['A3_Channel_Spatial'], input_tensor)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(raw_img.resize((224, 224)))
axes[0].set_title(f"Input: {idx_to_class[rep_label].capitalize()}", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(raw_img.resize((224, 224)))
axes[1].imshow(cam_baseline, cmap='jet', alpha=0.55)
axes[1].set_title("Baseline ResNet50 Grad-CAM", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(raw_img.resize((224, 224)))
axes[2].imshow(cam_attention, cmap='jet', alpha=0.55)
axes[2].set_title("CBAM (P3) Grad-CAM", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot09_gradcam_baseline_vs_attention.png"), dpi=300)
plt.close()
print("Saved plot09_gradcam_baseline_vs_attention.png")

# Plot 10: Spatial Attention Maps at P2, P3, P4 (Section 29, 31.13)
# Extract spatial attention map at P2 (28x28), P3 (14x14), P4 (7x7)
def get_spatial_map_at_stage(model_name):
    m = trained_models[model_name]
    m.eval()
    with torch.no_grad():
        _ = m(input_tensor)
    if hasattr(m, 'att_P2') and isinstance(m.att_P2, CBAMAttention):
        s = m.att_P2.sa.last_map[0, 0].numpy()
    elif hasattr(m, 'att_P3') and isinstance(m.att_P3, CBAMAttention):
        s = m.att_P3.sa.last_map[0, 0].numpy()
    elif hasattr(m, 'att_P4') and isinstance(m.att_P4, CBAMAttention):
        s = m.att_P4.sa.last_map[0, 0].numpy()
    else:
        s = np.zeros((14, 14))
    resized = Image.fromarray((s * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.array(resized) / 255.0

map_P2 = get_spatial_map_at_stage('B2_Position_P2')
map_P3 = get_spatial_map_at_stage('B3_Position_P3')
map_P4 = get_spatial_map_at_stage('B4_Position_P4')

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
axes[0].imshow(raw_img.resize((224, 224)))
axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(map_P2, cmap='inferno')
axes[1].set_title("$M_s^{P2}$ (conv3_x, $28\\times28$)\nLow-level edges/textures", fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(map_P3, cmap='inferno')
axes[2].set_title("$M_s^{P3}$ (conv4_x, $14\\times14$)\nIntermediate flower parts", fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(map_P4, cmap='inferno')
axes[3].set_title("$M_s^{P4}$ (conv5_x, $7\\times7$)\nHigh-level semantic regions", fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot10_attention_maps_by_position.png"), dpi=300)
plt.close()
print("Saved plot10_attention_maps_by_position.png")

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
    plt.scatter(x_val, y_val, s=130, zorder=5)
    plt.annotate(labels_map[name], (x_val, y_val), textcoords="offset points", xytext=(8, 4), fontsize=10, weight='semibold')

plt.title("Accuracy versus Total Model Parameters (Trade-off)", fontsize=13, fontweight='bold')
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
    plt.scatter(x_val, y_val, s=130, zorder=5)
    plt.annotate(labels_map[name], (x_val, y_val), textcoords="offset points", xytext=(8, 4), fontsize=10, weight='semibold')

plt.title("Macro F1-Score versus Inference Time per Image", fontsize=13, fontweight='bold')
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

plt.figure(figsize=(9, 4.8))
plt.bar(x - width/2, accs, width, label='Accuracy (%)', color='#3274a1', edgecolor='black')
plt.bar(x + width/2, f1s, width, label='Macro F1 (%)', color='#e1812c', edgecolor='black')
plt.title("Comparison of Attention Types (Fixed at Stage $P_3$)", fontsize=13, fontweight='bold')
plt.xlabel("Attention Mechanism", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(x, display_types)
plt.ylim(min(accs + f1s) - 5, 100)
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
plt.figure(figsize=(9.5, 4.8))
plt.bar(x - width/2, pos_accs, width, label='Accuracy (%)', color='#2ca02c', edgecolor='black')
plt.bar(x + width/2, pos_f1s, width, label='Macro F1 (%)', color='#9467bd', edgecolor='black')
plt.title("Influence of Attention Insertion Positions in ResNet50", fontsize=13, fontweight='bold')
plt.xlabel("Insertion Configuration", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(x, display_pos)
plt.ylim(min(pos_accs + pos_f1s) - 5, 100)
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
plt.ylim(min(k_accs + k_f1s) - 5, 100)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot15_spatial_kernel_ablation.png"), dpi=300)
plt.close()
print("Saved plot15_spatial_kernel_ablation.png")

# Plot 16: Qualitative Comparison across 3 correctly and 2 incorrectly classified images (Section 30, 32)
# Find correctly and misclassified images using baseline
base_preds = all_results['A0_Baseline']['preds']
base_targets = all_results['A0_Baseline']['targets']

correct_indices = [i for i, (p, t) in enumerate(zip(base_preds, base_targets)) if p == t]
incorrect_indices = [i for i, (p, t) in enumerate(zip(base_preds, base_targets)) if p != t]

selected_indices = correct_indices[:3] + incorrect_indices[:2]
fig, axes = plt.subplots(5, 6, figsize=(18, 15))
column_headers = ['Input Image', 'Baseline Grad-CAM', 'SE Attention', 'ECA Attention', 'CBAM Attention', 'Self-Attention']

models_to_compare = [
    ('A0_Baseline', compute_gradcam),
    ('A1_Channel', compute_gradcam),
    ('E1_ECA', compute_gradcam),
    ('A3_Channel_Spatial', compute_gradcam),
    ('A4_Self_Attention', compute_gradcam)
]

for row_idx, sample_idx in enumerate(selected_indices):
    img_path, true_lbl = test_subset[sample_idx]
    raw_sample = Image.open(img_path).convert('RGB')
    tensor_sample = data_transforms['val'](raw_sample).unsqueeze(0).to(device)
    
    # Input Image
    axes[row_idx, 0].imshow(raw_sample.resize((224, 224)))
    status = "Correct" if row_idx < 3 else "Misclassified"
    axes[row_idx, 0].set_ylabel(f"{status}\n({idx_to_class[true_lbl].capitalize()})", fontsize=11, fontweight='bold')
    axes[row_idx, 0].set_xticks([])
    axes[row_idx, 0].set_yticks([])
    if row_idx == 0:
        axes[row_idx, 0].set_title(column_headers[0], fontsize=12, fontweight='bold')

    for col_idx, (m_name, cam_fn) in enumerate(models_to_compare):
        m = trained_models[m_name]
        cam = cam_fn(m, tensor_sample)
        ax = axes[row_idx, col_idx + 1]
        ax.imshow(raw_sample.resize((224, 224)))
        ax.imshow(cam, cmap='jet', alpha=0.55)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(column_headers[col_idx + 1], fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "plot16_qualitative_multi_image_comparison.png"), dpi=300)
plt.close()
print("Saved plot16_qualitative_multi_image_comparison.png")

print("\nAll 16 required plots generated and saved successfully!")
