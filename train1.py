import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.distributions import Categorical

# =============================================================
# 1. CONFIGURATION
# =============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4

DATASET_ROOT = "/kaggle/input/icad-data/dataset_splited"
OUTPUT_DIR = "/kaggle/working/ppo"
METRICS_DIR = f"{OUTPUT_DIR}/metrics_images"
os.makedirs(METRICS_DIR, exist_ok=True)

# =============================================================
# 2. DATASET CLASS (Restored)
# =============================================================
class ICADDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load Image
        img = Image.open(self.img_paths[idx]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Load Mask
        mask = Image.open(self.mask_paths[idx]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        mask_np = (np.array(mask) > 127).astype(np.float32)
        
        # Convert to Tensors
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        return img_tensor, mask_tensor

# =============================================================
# 3. MODEL ARCHITECTURE
# =============================================================
class PPOUnetSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(encoder_name="resnet50", in_channels=1, classes=1, activation="sigmoid")
        self.rl_backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.actor = nn.Linear(16, 3) 
        self.critic = nn.Linear(16, 1)

    def forward(self, x): return self.unet(x)
    
    def get_ppo_action(self, mask):
        feat = self.rl_backbone(mask)
        return torch.softmax(self.actor(feat), dim=-1), self.critic(feat)

# =============================================================
# 4. METRIC PLOTTING FUNCTIONS
# =============================================================
def save_performance_plots(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(f"{METRICS_DIR}/loss_curve.png")
    plt.close()

    # 2. Dice & IoU Scores
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_dice'], 'g-', label='Val Dice')
    plt.plot(epochs, history['val_iou'], 'm-', label='Val IoU')
    plt.title('Segmentation Accuracy Metrics')
    plt.xlabel('Epochs'); plt.ylabel('Score')
    plt.legend(); plt.grid(True)
    plt.savefig(f"{METRICS_DIR}/accuracy_metrics.png")
    plt.close()

    # 3. Final Summary Table Image
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table_data = [
        ["Metric", "Value"],
        ["Best Dice", f"{max(history['val_dice']):.4f}"],
        ["Best IoU", f"{max(history['val_iou']):.4f}"],
        ["Final Loss", f"{history['train_loss'][-1]:.4f}"]
    ]
    ax.table(cellText=table_data, loc='center', cellLoc='center')
    plt.title("Performance Summary")
    plt.savefig(f"{METRICS_DIR}/summary_table.png")
    plt.close()

# =============================================================
# 5. TRAINING EXECUTION
# =============================================================
def train():
    all_imgs = sorted(glob(os.path.join(DATASET_ROOT, "*/train/images/*")))
    all_masks = sorted(glob(os.path.join(DATASET_ROOT, "*/train/masks/*")))
    
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {DATASET_ROOT}. Check your paths.")

    train_i, val_i, train_m, val_m = train_test_split(all_imgs, all_masks, test_size=0.15, random_state=42)

    train_loader = DataLoader(ICADDataset(train_i, train_m), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ICADDataset(val_i, val_m), batch_size=1)
    
    model = PPOUnetSegmentor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    dice_loss_fn = smp.losses.DiceLoss(mode='binary')
    bce_loss_fn = nn.BCELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        for img, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model(img)
            loss = 0.5 * dice_loss_fn(pred, target) + 0.5 * bce_loss_fn(pred, target)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
            
        model.eval()
        v_loss, v_dice, v_iou = 0, 0, 0
        with torch.no_grad():
            for img, target in val_loader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                pred = model(img)
                v_loss += (0.5 * dice_loss_fn(pred, target) + 0.5 * bce_loss_fn(pred, target)).item()
                
                p = (pred > 0.5).float()
                t = (target > 0.5).float()
                
                inter = (p * t).sum()
                union = p.sum() + t.sum() - inter
                v_dice += (2 * inter + 1e-7) / (p.sum() + t.sum() + 1e-7)
                v_iou += (inter + 1e-7) / (union + 1e-7)
        
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(val_loader))
        history['val_dice'].append((v_dice/len(val_loader)).item())
        history['val_iou'].append((v_iou/len(val_loader)).item())

        print(f"Epoch {epoch} | Loss: {history['train_loss'][-1]:.4f} | Dice: {history['val_dice'][-1]:.4f}")

    save_performance_plots(history)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/ppo_unet_model.pth")
    return history

if __name__ == "__main__":
    train_history = train()
    print(f"Training complete. Metrics saved in: {METRICS_DIR}")