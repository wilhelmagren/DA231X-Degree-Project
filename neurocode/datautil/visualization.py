"""

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 28-01-2022
"""
import matplotlib.pyplot as plt


def plot_history_acc_loss(history, marker='d', ms=5, alpha=.7, fpath='history.png'):
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = ax1.twinx()
    ax1.plot(history['tloss'], ls='-', marker=marker, ms=ms, alpha=alpha, color='tab:blue', label='training loss')
    ax1.plot(history['vloss'], ls=':', marker=marker, ms=ms, alpha=alpha, color='tab:blue', label='validation loss')
    ax2.plot(history['tacc'], ls='-', marker=marker, ms=ms, alpha=alpha, color='tab:orange', label='training acc')
    ax2.plot(history['vacc'], ls=':', marker=marker, ms=ms, alpha=alpha, color='tab:orange', label='validation acc')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Accuracy [%]', color='tab:orange')
    ax1.set_xlabel('Epoch')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.clf()

