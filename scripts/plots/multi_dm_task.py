import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path


# load fonts/rc file for plotting
font_path = Path(__file__).parent.parent.parent / "fonts/Arial.ttf"
matplotlib.font_manager.fontManager.addfont(font_path)

rc_path = Path(__file__).parent.parent.parent / ".matplotlibrc"


# load datasets
dataset_dir = Path(__file__).parent.parent / "approx_rnn" / "datasets"
dataset = np.load(dataset_dir / "mean_1.npz")


# plot inputs and outputs
input_data = dataset["inputs"]
output_data = dataset["targets"]
trial_num = 4

with matplotlib.rc_context(fname=rc_path):
    fig, ax = plt.subplots(2, 1, figsize=(2, 1.8), sharex=True)
    ax[0].plot(input_data[trial_num, :, 3], color="blue", label="stim1")
    ax[0].legend(loc="lower right")
    ax[0].spines['bottom'].set_visible(False)
    ax[0].set_ylim(-1, 1)
    ax[0].set_xticks([])

    ax[1].plot(input_data[trial_num, :, 4], color="orange", label="stim2")
    ax[1].legend(loc="upper right")
    ax[1].set_ylim(-1, 1)

    ax[1].set_xlabel("time")

    plt.tight_layout()
    plt.savefig("inputs.png")
    plt.close()

with matplotlib.rc_context(fname=rc_path):
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.2))
    ax.plot(output_data[trial_num, :, 0], color="black", label="target")
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])

    ax.set_xlabel("time")

    plt.tight_layout()
    plt.savefig("outputs.png")
    plt.close()