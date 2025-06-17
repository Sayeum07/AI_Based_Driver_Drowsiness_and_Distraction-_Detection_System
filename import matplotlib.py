import matplotlib.pyplot as plt
import numpy as np

# Function to add values on top of bars
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}' + ('%' if height > 1 else ''),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# -------------------------------
# 🔷 1. ResNet50 Fine-tuned Model
# -------------------------------
metrics_resnet = ['Accuracy', 'Precision', 'Recall', 'Loss']
training_resnet = [98.58, 98.58, 98.58, 0.0424]
validation_resnet = [92.70, 92.70, 92.70, 0.1749]

x = np.arange(len(metrics_resnet))
width = 0.35

fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(x - width/2, training_resnet, width, label='Training', color='#4CAF50')
bars2 = ax1.bar(x + width/2, validation_resnet, width, label='Validation', color='#2196F3')

ax1.set_ylabel('Percentage / Loss')
ax1.set_title('ResNet50 – Training vs Validation Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_resnet)
ax1.legend()
add_labels(bars1, ax1)
add_labels(bars2, ax1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('resnet50_model_performance_graph.png')  # Save the figure

# -------------------------------
# 🔶 2. MobileNet 3-Layer Model
# -------------------------------
metrics_mobilenet = ['Accuracy', 'Precision', 'Recall', 'Loss']
training_mobilenet = [91.76, 91.76, 91.76, 0.5256]
validation_mobilenet = [88.32, 88.32, 88.32, 0.6178]

x_mobilenet = np.arange(len(metrics_mobilenet))

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars1_m = ax2.bar(x_mobilenet - width/2, training_mobilenet, width, label='Training', color='#FF9800')
bars2_m = ax2.bar(x_mobilenet + width/2, validation_mobilenet, width, label='Validation', color='#03A9F4')

ax2.set_ylabel('Percentage / Loss')
ax2.set_title('MobileNet – Training vs Validation Metrics')
ax2.set_xticks(x_mobilenet)
ax2.set_xticklabels(metrics_mobilenet)
ax2.legend()
add_labels(bars1_m, ax2)
add_labels(bars2_m, ax2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('mobilenet_model_performance_graph.png')  # Save the figure

plt.show()
