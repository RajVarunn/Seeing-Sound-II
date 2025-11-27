#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Training data extracted from your log
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Loss data
total_loss = [0.7396, 0.5860, 0.5249, 0.4801, 0.4483, 0.4235, 0.4060, 0.3890, 0.3754, 0.3604]
clap_loss = [0.3602, 0.2445, 0.2065, 0.1753, 0.1537, 0.1372, 0.1280, 0.1182, 0.1104, 0.0999]
sd_loss = [0.5595, 0.4638, 0.4217, 0.3924, 0.3715, 0.3549, 0.3420, 0.3299, 0.3202, 0.3105]

# Similarity scores
clap_sim = [0.331, 0.369, 0.384, 0.396, 0.409, 0.425, 0.439, 0.441, 0.447, 0.454]
sd_sim = [0.684, 0.747, 0.773, 0.791, 0.803, 0.813, 0.820, 0.827, 0.833, 0.839]

# CLIP evaluation scores (only at epochs 2, 4, 6, 8, 10)
clip_epochs = [2, 4, 6, 8, 10]
clip_scores = [19.5569, 21.2103, 22.4712, 21.3244, 23.0220]

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Audio2Image Training Progress (MLP Only Model)', fontsize=16, fontweight='bold')

# Plot 1: Loss curves
ax1.plot(epochs, total_loss, 'b-o', label='Total Loss', linewidth=2)
ax1.plot(epochs, clap_loss, 'r-s', label='CLAP Loss', linewidth=2)
ax1.plot(epochs, sd_loss, 'g-^', label='SD Loss', linewidth=2)
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
summary_text = f"""Training Summary:
• Dataset: 18,643 samples (16,778 train, 1,865 val)
• Batch Size: 4
• Epochs: 10
• Best CLIP Score: {max(clip_scores):.4f} (Epoch {clip_epochs[np.argmax(clip_scores)]})
• Final Total Loss: {total_loss[-1]:.4f}
• Final CLAP Loss: {clap_loss[-1]:.4f}
• Final SD Loss: {sd_loss[-1]:.4f}
• Final CLAP Sim: {clap_sim[-1]:.3f}
• Final SD Sim: {sd_sim[-1]:.3f}

Loss Reduction:
• Total: {((total_loss[0] - total_loss[-1]) / total_loss[0] * 100):.1f}% decrease
• CLAP: {((clap_loss[0] - clap_loss[-1]) / clap_loss[0] * 100):.1f}% decrease
• SD: {((sd_loss[0] - sd_loss[-1]) / sd_loss[0] * 100):.1f}% decrease"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Loss curves saved as: training_loss_curves.png")
print(f"📊 Training completed with {((total_loss[0] - total_loss[-1]) / total_loss[0] * 100):.1f}% total loss reduction")
print(f"🎯 Best CLIP score: {max(clip_scores):.4f} at epoch {clip_epochs[np.argmax(clip_scores)]}")