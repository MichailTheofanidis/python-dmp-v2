# %%

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_trajectory_data(folder, traj_id):

    data_path = project_root / "data" / folder / f"traj{traj_id}.mat"

    print(f"Loading trajectory data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find trajectory file: {data_path}")

    mat_data = sio.loadmat(data_path)
    demo = mat_data["demodata"][0, 0]

    # Build a DataFrame
    df = pd.DataFrame({
        "time": np.asarray(demo["t"]).ravel(),
        "x": np.asarray(demo["x"]).ravel(),
        "y": np.asarray(demo["y"]).ravel(),
        "z": np.asarray(demo["z"]).ravel(),
    })

    return df


if __name__ == "__main__":

    folder = ["flight1", "flight2"]
    traj_id = [1, 2]

    df = load_trajectory_data(folder[0], traj_id[0])

    print(df.head())

    # NumPy arrays for algorithms
    time = df["time"].to_numpy()

    trajectory = df[["x", "y", "z"]].to_numpy()

    print(f"\nLoaded {folder[0]}/traj{traj_id[0]}.mat")
    print(f"Time samples: {len(time)}")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"First point: {trajectory[0]}")


# %%
