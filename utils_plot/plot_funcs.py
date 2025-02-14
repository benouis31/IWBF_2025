"""
This code from @UNIROMA3
performance visualization functions.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_roc(fpr, tpr, figure_name="roc.png"):
    plt.switch_backend('Agg')

    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.8f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    fig.savefig(figure_name, dpi=fig.dpi)


def plot_DET_with_EERr(far, frr, far_optimum, frr_optimum, figure_name):
    """ Plots a DET curve with the most suitable operating point based on threshold values"""
    fig = plt.figure()
    lw = 2
    # Plot the DET curve based on the FAR and FRR values
    EER = float((far_optimum + frr_optimum) / 2)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.plot(far, frr, color='red', linewidth=lw, label='DET Curve (EER = %0.8f)' % EER)
    # Plot the optimum point on the DET Curve
    plt.plot(far_optimum, frr_optimum, "ko", label="Suitable Operating Point")

    plt.xlim([-0.01, 0.1])
    plt.ylim([-0.01, 0.1])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('Detection Error Tradeoff')
    plt.legend(loc="upper right")
    plt.grid(True)
    fig.savefig(figure_name, dpi=fig.dpi)
    
    




def plot_DET_with_EER(far, frr, far_optimum, frr_optimum, figure_name):
    """
    Plots a DET curve with the most suitable operating point based on threshold values.
    """
    fig = plt.figure()
    lw = 2

    # Calculate the true EER based on the crossover point
    abs_diff = np.abs(np.array(far) - np.array(frr))
    eer_index = np.argmin(abs_diff)
    EER = (far[eer_index] + frr[eer_index]) / 2

    # Plot the DET curve
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--', label='Chance Line')
    plt.plot(far, frr, color='red', linewidth=lw, label='DET Curve (EER = %0.8f)' % EER)
    plt.plot(far_optimum, frr_optimum, "ko", label="Suitable Operating Point")

    # Adjust axis limits dynamically
    plt.xlim([min(far) - 0.01, max(far) + 0.01])
    plt.ylim([min(frr) - 0.01, max(frr) + 0.01])

    # Customize ticks and gridlines
    plt.xticks(np.linspace(0, max(far) + 0.01, 6))
    plt.yticks(np.linspace(0, max(frr) + 0.01, 6))
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    # Labels and title
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('Detection Error Tradeoff')

    # Legend placement
    plt.legend(loc="lower right")

    # Save the figure
    fig.savefig(figure_name, dpi=fig.dpi)








def plot_densityb(distances, labels, figure_name):
    fig = plt.figure()
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == 0)
    p1 = sns.histplot(distances[pos_index], kde=True, stat="density", bins=50, color="r", label="Genuine")
    p1 = sns.histplot(distances[neg_index], kde=True, stat="density", bins=50, color="b", label="Impostor")
    
    
    
    #p1 = sns.distplot(distances[pos_index], kde=True, norm_hist=False, bins=50, color="r", label="Genuine")
    #p1 = sns.distplot(distances[neg_index], kde=True, norm_hist=False, bins=50, color="b", label="Impostor", )
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(str, locs * 0.01)))
    plt.ylabel('Density Distribution [%]', fontsize=18)
    plt.xlabel('Similarity Distance', fontsize=18)
    fig.savefig(figure_name)







def plot_density(distances, labels, figure_name):
    fig = plt.figure()

    # Get the indices for genuine and impostor labels
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == 0)
    
    # Plot Genuine (Red)
    sns.histplot(distances[pos_index], stat="density", bins=50, color="r", label="Genuine", alpha=0.6)
    
    # Plot Impostor (Blue)
    sns.histplot(distances[neg_index], stat="density", bins=50, color="b", label="Impostor", alpha=0.6)
    
    # Set the y-axis ticks
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(str, locs * 0.01)))
    
    # Set labels for axes
    plt.ylabel('Density Distribution [%]', fontsize=18)
    plt.xlabel('Similarity Distance', fontsize=18)

    # Add legend
    plt.legend()

    # Save the figure
    fig.savefig(figure_name)
    plt.close()



















def plot_densitys(distances, labels, figure_name):
    # Separate the genuine and impostor distances
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == 0)
    genuine_distances = distances[pos_index]
    impostor_distances = distances[neg_index]

    # Center the distributions (optional shift point can be customized)
    shift_point = 0.75  # Example center point
    genuine_distances_centered = genuine_distances - np.mean(genuine_distances) + shift_point
    impostor_distances_centered = impostor_distances - np.mean(impostor_distances) + shift_point

    # Plot the centered distributions
    fig = plt.figure()
    p1 = sns.histplot(genuine_distances_centered, kde=True, stat="density", bins=50, color="r", label="Genuine")
    p1 = sns.histplot(impostor_distances_centered, kde=True, stat="density", bins=50, color="b", label="Impostor")

    # Adjust y-axis and labels
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(str, locs * 0.01)))
    plt.ylabel('Density Distribution [%]', fontsize=18)
    plt.xlabel('Similarity Distance', fontsize=18)
    
    # Add legend
    plt.legend(fontsize=14)
    
    # Save the figure
    fig.savefig(figure_name)



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_densityd(distances, labels, figure_name):
    # Separate the genuine and impostor distances
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == 0)
    genuine_distances = distances[pos_index]
    impostor_distances = distances[neg_index]

    # Center the distributions (optional shift point can be customized)
    shift_point = 0.75  # Example center point
    genuine_distances_centered = genuine_distances - np.mean(genuine_distances) + shift_point
    impostor_distances_centered = impostor_distances - np.mean(impostor_distances) + shift_point

    # Plot the centered distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(genuine_distances_centered, kde=True, stat="density", bins=50,
                 color="r", label="Genuine", alpha=0.5, ax=ax)
    sns.histplot(impostor_distances_centered, kde=True, stat="density", bins=50,
                 color="b", label="Impostor", alpha=0.5, ax=ax)

    # Adjust y-axis labels for better visibility
    plt.ylabel('Density Distribution', fontsize=14)
    plt.xlabel('Similarity Distance', fontsize=14)
    
    # Add legend
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(figure_name)





from matplotlib.ticker import FuncFormatter

def plot_densityn(distances, labels, figure_name):
    # Separate the genuine and impostor distances
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == 0)
    genuine_distances = distances[pos_index]
    impostor_distances = distances[neg_index]

    # Center the distributions (optional shift point can be customized)
    shift_point = 0.75
    genuine_distances_centered = genuine_distances - np.mean(genuine_distances) + shift_point
    impostor_distances_centered = impostor_distances - np.mean(impostor_distances) + shift_point

    # Plot the distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(genuine_distances_centered, stat="density", bins=100, color="r", alpha=0.5, label="Genuine", ax=ax)
    sns.histplot(impostor_distances_centered, stat="density", bins=100, color="b", alpha=0.5, label="Impostor", ax=ax)

    # Add density curves
    sns.kdeplot(genuine_distances_centered, color="r", linewidth=2, ax=ax)
    sns.kdeplot(impostor_distances_centered, color="b", linewidth=2, ax=ax)

    # Adjust axis limits
    ax.set_xlim(0.4, 1.1)
    ax.set_ylim(0, None)  # Automatically adjusts to fit the data

    # Set labels and legend
    ax.set_xlabel("Similarity Distance", fontsize=14)
    ax.set_ylabel("Density Distribution", fontsize=14)
    ax.legend(fontsize=12)

    # Format y-axis to prevent scientific notation
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Save the figure
    plt.tight_layout()
    fig.savefig(figure_name)
