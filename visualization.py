import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, name, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_roc_curve(roc_data, save_path):
    plt.figure()
    for fpr, tpr, roc_auc, name in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def merge_images(image_paths, save_path, grid_size=(2, 3)):
    images = [Image.open(img) for img in image_paths]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    total_width = max_width * grid_size[1]
    total_height = max_height * grid_size[0]
    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    for index, image in enumerate(images):
        x_offset = (index % grid_size[1]) * max_width
        y_offset = (index // grid_size[1]) * max_height
        new_image.paste(image, (x_offset, y_offset))
    new_image.save(save_path)
    print(f'Combined image saved to {save_path}')
