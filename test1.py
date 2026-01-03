import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp

# -----------------------------
# CONFIGURATION
# -----------------------------
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384  # Matches your training IMG_SIZE

# Kaggle-specific paths
DATASET_ROOT = r"D:\28-12-2025(RLUnet)\dataset_splited"
MODEL_PATH   = r"D:\28-12-2025(RLUnet)\miccai\ppo_unet_model.pth"
OUTPUT_DIR   = r"D:\28-12-2025(RLUnet)\miccai\detailed_inference_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# MODEL ARCHITECTURE (Matches PPO Training)
# =============================================================
class SegmentationAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard U-Net Backbone
        self.unet = smp.Unet(encoder_name="resnet50", in_channels=1, classes=1, activation="sigmoid")
        
        # PPO Refiner Head (Stored to match state_dict, even if unused in simple inference)
        self.rl_backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.actor = nn.Linear(16, 3) 
        self.critic = nn.Linear(16, 1)

    def forward(self, x):
        return self.unet(x)

# =============================================================
# DETAILED VISUALIZATION (NumPy Blending)
# =============================================================
def save_detailed_viz(raw_np, gt_mask, pred_mask, save_path, image_id):
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle(f"RL-Unet Clinical Analysis Report | ID: {image_id}", fontsize=20, fontweight='bold', y=0.95)

    # Prepare overlay background
    base_rgb = (raw_np * 255).astype(np.uint8)
    overlay = cv2.cvtColor(base_rgb, cv2.COLOR_GRAY2RGB).astype(np.float32)
    
    alpha = 0.4 
    
    # Apply Ground Truth (Green: 0, 255, 0)
    gt_idx = gt_mask == 1
    overlay[gt_idx, 1] = overlay[gt_idx, 1] * (1 - alpha) + 255 * alpha

    # Apply Prediction (Red: 255, 0, 0)
    pred_idx = pred_mask == 1
    overlay[pred_idx, 0] = overlay[pred_idx, 0] * (1 - alpha) + 255 * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    panels = [raw_np, gt_mask, pred_mask, overlay]
    titles = ["Input CT Image", "Ground Truth (GT)", "RL-Unet Prediction", "Overlay Analysis (GT:G, Pred:R)"]
    
    for i in range(4):
        axes[i].imshow(panels[i], cmap='gray' if i < 3 else None)
        axes[i].set_title(titles[i], fontsize=15, pad=10)
        axes[i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

# =============================================================
# RUN INFERENCE
# =============================================================
def run_inference():
    # 1. Initialize and load model
    model = SegmentationAgent().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        # weights_only=False used here to ensure compatibility with smp models
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Successfully loaded RL-Unet from {MODEL_PATH}")
    else:
        print(f"ERROR: Model not found. Please check path: {MODEL_PATH}")
        return

    model.eval()

    # 2. Gather test images
    test_image_paths = sorted(glob(os.path.join(DATASET_ROOT, "*/test/images/*")))
    print(f"Processing {len(test_image_paths)} test images...")

    with torch.no_grad():
        for img_path in tqdm(test_image_paths):
            file_name = os.path.basename(img_path)
            image_id = os.path.splitext(file_name)[0]
            
            category = "pos" if "positive" in img_path.lower() else "neg"
            unique_id = f"{category}_{image_id}"
            
            # Load and preprocess
            img_raw = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
            raw_np = np.array(img_raw, dtype=np.float32) / 255.0
            img_t = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

            # Find matching mask
            mask_dir_path = img_path.replace("images", "masks")
            mask_search = glob(f"{os.path.dirname(mask_dir_path)}/{image_id}*")
            mask_path = mask_search[0] if mask_search else None

            # Generate Prediction
            pred_score = model(img_t).squeeze().cpu().numpy()
            pred_bin = (pred_score > 0.5).astype(np.uint8)

            # Load Ground Truth
            if mask_path:
                gt_img = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                gt_bin = (np.array(gt_img) > 127).astype(np.uint8)
            else:
                gt_bin = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

            # Save report
            save_path = os.path.join(OUTPUT_DIR, f"{unique_id}_analysis.png")
            save_detailed_viz(raw_np, gt_bin, pred_bin, save_path, unique_id)

if __name__ == "__main__":
    run_inference()