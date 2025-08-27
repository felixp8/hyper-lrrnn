import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

from pathlib import Path


# load fonts/rc file for plotting
font_path = Path(__file__).parent.parent.parent / "fonts/Arial.ttf"
matplotlib.font_manager.fontManager.addfont(font_path)

rc_path = Path(__file__).parent.parent.parent / ".matplotlibrc"


# define grid
task_signals = ["mean", "integral", "time"]
task_targets = ["1", "2", "sum", "diff"]

# load datasets
mixture_dir = Path(__file__).parent.parent / "approx_rnn" / "mixture_params"

for sig, tgt in itertools.product(task_signals, task_targets):
    mixture = np.load(mixture_dir / f"{sig}_{tgt}_mixture1.npz")
    orig_losses = mixture["orig_accs"]
    model_1m_losses = mixture["model_accs"]
    random_1m_losses = mixture["random_accs"]

    mixture = np.load(mixture_dir / f"{sig}_{tgt}_mixture2.npz")
    model_2m_losses = mixture["model_accs"]
    random_2m_losses = mixture["random_accs"]

    data = [orig_losses, model_1m_losses, model_2m_losses]
    labels = ['Original', 'Gaussian', 'MoG (n=2)']

    with matplotlib.rc_context(fname=rc_path):
        fig, axs = plt.subplots(1, 1, figsize=(2.5, 2))
        sns.violinplot(data=data, fill=False)
        axs.plot([0.7, 1.3], [np.median(random_1m_losses)] * 2, color='black', linestyle='--')
        axs.plot([1.7, 2.3], [np.median(random_2m_losses)] * 2, color='black', linestyle='--')
        # for i, losses in enumerate(data):
        #     axs.scatter([i + np.random.randn() / 10 for _ in range(len(losses))], losses, color='black', s=5, alpha=0.3)
        axs.set_xticks(range(len(labels)))
        axs.set_xticklabels(labels)
        axs.set_ylabel("accuracy")
        axs.set_ylim(0.3, 1.1)
        # axs.set_ylim(0.3, 0.9)
        plt.tight_layout()
        plt.savefig(f'{sig}_{tgt}.png')
        plt.show()