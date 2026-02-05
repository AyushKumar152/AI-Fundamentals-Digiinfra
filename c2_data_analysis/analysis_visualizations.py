import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_analysis_visualizations(distribution, cooccurrence_matrix, output_folder="analysis_results"):
    os.makedirs(output_folder, exist_ok=True)

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values())
    plt.title("Class Pixel Distribution (%)")
    plt.xlabel("Class ID")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "class_distribution.png"))
    plt.close()

    # 2. Co-occurrence Heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(cooccurrence_matrix, annot=True, cmap="YlGnBu", fmt="d")
    plt.title("Class Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "cooccurrence_heatmap.png"))
    plt.close()
