from pathlib import Path
import sys

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from assets import plots
from src.dmp import Dmp
from lerobot.datasets.lerobot_dataset import LeRobotDataset

project_root = Path(__file__).resolve().parents[1]


ds = LeRobotDataset(
    repo_id="xerous/lift_white_color_object_teleop_20260909_190957",
    root=r"C:\Users\theofanidism\PycharmProjects\Python-DMP\data\gello_fr3_dmp"
)

print(ds.num_episodes, ds.num_frames, ds.fps)
print(ds.meta.features.keys())
frame = ds[0]   # dict of tensors (images decoded from video)


EP = 7  # episode to plot

cols = ds.hf_dataset.select_columns(["ee_pose", "joint_action", "timestamp", "episode_index"]).with_format("numpy")
mask = np.asarray(cols["episode_index"]) == EP
ee = np.stack(cols["ee_pose"])[mask]
q = np.stack(cols["joint_action"])[mask]
t = np.asarray(cols["timestamp"])[mask]
order = np.argsort(t)
ee, q, t = ee[order], q[order], t[order]


n_joints = q.shape[1]

fig, axes = plt.subplots(
    n_joints, 1,
    figsize=(12, 2.5 * n_joints),
    sharex=True
)

for i in range(n_joints):
    axes[i].plot(t, q[:, i], linewidth=1.5)
    axes[i].set_ylabel(f"Joint {i+1}\n[rad]")
    axes[i].grid(alpha=0.3)

axes[0].set_title(f"Joint angles, episode {EP}")
axes[-1].set_xlabel("time [s]")

# Initialize the DMP class
my_dmp = Dmp()

s = my_dmp.phase(t)
psi = my_dmp.basis_functions(s)

w, f = my_dmp.imitate(q, t, s, psi)

[p, dp, ddp] = my_dmp.generate(w, q[0], q[-1], t, s, psi)

fig, axes = plt.subplots(
    n_joints, 1,
    figsize=(12, 2.5 * n_joints),
    sharex=True
)

for i in range(n_joints):
    axes[i].plot(t, q[:, i], label="Original (q)", linewidth=1.5)
    axes[i].plot(t, p[:, i], "--", label="DMP (p)", linewidth=1.5)

    axes[i].set_ylabel(f"Joint {i+1}\n[rad]")
    axes[i].grid(alpha=0.3)
    axes[i].legend()

axes[0].set_title(f"DMP imitation vs. original trajectory, episode {EP}")
axes[-1].set_xlabel("time [s]")

fig, axes = plt.subplots(
    n_joints, 1,
    figsize=(12, 2.5 * n_joints),
    sharex=True
)

for i in range(n_joints):
    for j in range(psi.shape[0]):
        axes[i].plot(
            t,
            psi[j, :] * w[j, i] * s,
            linewidth=1.2,
            label=f"Gaussian {j+1}"
        )

    axes[i].set_ylabel(f"Joint {i+1}\n[rad]")
    axes[i].grid(alpha=0.3)

axes[0].set_title(f"Weighted Gaussian Activations, episode {EP}")
axes[-1].set_xlabel("time [s]")

plt.tight_layout()
plt.show()