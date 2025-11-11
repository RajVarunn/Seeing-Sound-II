#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Training data extracted from UNet end-to-end training log
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Loss data
total_loss = [0.8739, 0.7207, 0.6598, 0.6129, 0.5864, 0.5603, 0.5405, 0.5208, 0.5094, 0.4965]
clap_loss = [0.3498, 0.2357, 0.2058, 0.1707, 0.1566, 0.1379, 0.1249, 0.1123, 0.1085, 0.1004]
sd_loss = [0.5617, 0.4646, 0.4215, 0.3923, 0.3716, 0.3555, 0.3422, 0.3303, 0.3196, 0.3100]
diff_loss = [0.1372, 0.1382, 0.1353, 0.1353, 0.1366, 0.1359, 0.1358, 0.1343, 0.1356, 0.1363]

# Similarity scores
clap_sim = [0.341, 0.378, 0.390, 0.410, 0.420, 0.432, 0.439, 0.448, 0.459, 0.462]
sd_sim = [0.683, 0.747, 0.774, 0.791, 0.803, 0.813, 0.820, 0.827, 0.834, 0.839]

# CLIP evaluation scores (at epochs 2, 4, 6, 8, 10)
clip_epochs = [2, 4, 6, 8, 10]
clip_scores = [20.394, 20.511, 19.939, 22.311, 19.938]

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Audio2Image Training Progress (UNet End-to-End Model)', fontsize=16, fontweight='bold')

# Plot 1: Loss curves
ax1.plot(epochs, total_loss, 'b-o', label='Total Loss', linewidth=2)
ax1.plot(epochs, clap_loss, 'r-s', label='CLAP Loss', linewidth=2)
ax1.plot(epochs, sd_loss, 'g-^', label='SD Loss', linewidth=2)
ax1.plot(epochs, diff_loss, 'm-d', label='Diffusion Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Similarity scores
ax2.plot(epochs, clap_sim, 'r-s', label='CLAP Similarity', linewidth=2)
ax2.plot(epochs, sd_sim, 'g-^', label='SD Similarity', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Similarity Score')
ax2.set_title('Similarity Scores')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: CLIP evaluation scores
ax3.plot(clip_epochs, clip_scores, 'purple', marker='o', linewidth=3, markersize=8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('CLIP Score')
ax3.set_title('CLIP Evaluation Scores')
ax3.grid(True, alpha=0.3)
# Highlight best score
best_idx = np.argmax(clip_scores)
ax3.plot(clip_epochs[best_idx], clip_scores[best_idx], 'ro', markersize=12, alpha=0.7)
ax3.annotate(f'Best: {clip_scores[best_idx]:.2f}', 
             xy=(clip_epochs[best_idx], clip_scores[best_idx]),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Plot 4: Training summary
ax4.axis('off')
summary_text = f"""Training Summary (UNet End-to-End):
• Dataset: 18,643 samples (16,778 train, 1,865 val)
• Batch Size: 4
• Epochs: 10
• Best CLIP Score: {max(clip_scores):.4f} (Epoch {clip_epochs[np.argmax(clip_scores)]})
• Final Total Loss: {total_loss[-1]:.4f}
• Final CLAP Loss: {clap_loss[-1]:.4f}
• Final SD Loss: {sd_loss[-1]:.4f}
• Final Diffusion Loss: {diff_loss[-1]:.4f}
• Final CLAP Sim: {clap_sim[-1]:.3f}
• Final SD Sim: {sd_sim[-1]:.3f}

Loss Reduction:
• Total: {((total_loss[0] - total_loss[-1]) / total_loss[0] * 100):.1f}% decrease
• CLAP: {((clap_loss[0] - clap_loss[-1]) / clap_loss[0] * 100):.1f}% decrease
• SD: {((sd_loss[0] - sd_loss[-1]) / sd_loss[0] * 100):.1f}% decrease
• Diffusion: {((diff_loss[0] - diff_loss[-1]) / diff_loss[0] * 100):.1f}% decrease"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('unet_training_loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ UNet loss curves saved as: unet_training_loss_curves.png")
print(f"📊 UNet training completed with {((total_loss[0] - total_loss[-1]) / total_loss[0] * 100):.1f}% total loss reduction")
print(f"🎯 Best CLIP score: {max(clip_scores):.4f} at epoch {clip_epochs[np.argmax(clip_scores)]}")
print(f"🔥 End-to-end training with diffusion loss: {((diff_loss[0] - diff_loss[-1]) / diff_loss[0] * 100):.1f}% change")