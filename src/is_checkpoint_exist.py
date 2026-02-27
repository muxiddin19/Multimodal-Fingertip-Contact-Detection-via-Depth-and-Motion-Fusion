import os

# Check if the ViT-S checkpoint exists
vits_path = r"D:/Codes/vscode/Pretrained_weights/depth_anything_v2_metric_hypersim_vits.pth"
vitb_path = r"D:/Codes/vscode/Pretrained_weights/depth_anything_v2/latest20.pth"

print("Checkpoint files:")
print(f"  ViT-S: {os.path.exists(vits_path)} - {vits_path}")
print(f"  ViT-B: {os.path.exists(vitb_path)} - {vitb_path}")