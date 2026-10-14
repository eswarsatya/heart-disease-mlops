import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# Colors
colors = {'blue': '#a8d8ea', 'green': '#aa96da', 'yellow': '#ffffd2', 'pink': '#fcbad3'}

# Boxes
boxes = [
    (0.5, 2, 1.2, 0.8, "User\nClient", colors['blue']),
    (2.5, 2, 1.5, 0.8, "GitHub Actions\nCI/CD", colors['yellow']),
    (4.5, 2, 1.2, 0.8, "Docker\nContainer", colors['green']),
    (6.5, 2, 1.5, 0.8, "Kubernetes\nDeployment", colors['green']),
    (4.5, 0.5, 1.2, 0.8, "FastAPI\n/predict", colors['pink']),
    (6.5, 0.5, 1.5, 0.8, "Prometheus\n/metrics", colors['pink']),
]

for x, y, w, h, text, color in boxes:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
ax.annotate("", xy=(4.5, 2.4), xytext=(1.7, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(6.5, 2.4), xytext=(6.0, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(5.1, 1.3), xytext=(5.1, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(6.5, 0.9), xytext=(5.7, 0.9), arrowprops=dict(arrowstyle="->", lw=1.5))

plt.title('Heart Disease MLOps - End-to-End Architecture', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('screenshots/architecture.png', dpi=300, bbox_inches='tight')
print("Saved screenshots/architecture.png")
plt.close()