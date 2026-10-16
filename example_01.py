from pathlib import Path
import sys

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from assets import plots
from src.dmp import Dmp

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_trajectory_file(folder, traj_id):
    """
    Load a single trajectory file.

    Returns:
        trajectory: Nx4 numpy array
                    columns = [t, x, y, z]
    """

    data_path = project_root / "data" / folder / f"traj{traj_id}.mat"

    print(f"Loading: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find trajectory file: {data_path}"
        )

    mat_data = sio.loadmat(data_path)

    demo = mat_data["demodata"][0, 0]

    t = np.asarray(demo["t"]).ravel()
    x = np.asarray(demo["x"]).ravel()
    y = np.asarray(demo["y"]).ravel()
    z = np.asarray(demo["z"]).ravel()

    # Nx4 matrix
    trajectory = np.column_stack((t, x, y, z))

    return trajectory

def load_flights(flights="all", traj_ids="all"):
    """
    Load multiple flights and trajectories.

    Parameters
    ----------
    flights : list or "all"
        Example:
            ["flight1"]
            ["flight1", "flight2"]
            "all"

    traj_ids : list or "all"
        Example:
            [1,2,3]
            [2,3,4]
            "all"

    Returns
    -------
    data : dictionary
        data["flight1"]["traj1"] = Nx4 array
    """
    data_folder = project_root / "data"

    # Determine flights
    if flights == "all":
        flights = [
            f.name for f in data_folder.iterdir()
            if f.is_dir()
        ]

    data = {}

    for flight in flights:

        flight_path = data_folder / flight

        if not flight_path.exists():
            raise FileNotFoundError(
                f"Flight folder does not exist: {flight_path}"
            )

        # Determine trajectory IDs
        if traj_ids == "all":

            files = list(flight_path.glob("traj*.mat"))

            ids = []

            for f in files:
                name = f.stem      # traj1
                ids.append(int(name.replace("traj", "")))

        else:
            ids = traj_ids


        data[flight] = {}

        for traj_id in sorted(ids):

            trajectory = load_trajectory_file(
                flight,
                traj_id
            )

            data[flight][f"traj{traj_id}"] = trajectory

    return data

if __name__ == "__main__":

    # Load the data
    data = load_flights(
        flights="all",
        traj_ids="all"
    )

    # Initialize DMP data structure
    dmp_data = {}

    for flight_name, flight_data in data.items():

        dmp_data[flight_name] = {}

        for traj_name in flight_data:
            dmp_data[flight_name][traj_name] = {
                "S": None,
                "PsV": None,
                "f": None,
                "w": None,
                "p": None
            }

    # Plot the data
    plots.plot_flight_trajectories( data,
        num_flights=2
    )

    plots.plot_3d_flight_trajectories( data,
        num_flights=2
    )

    # Initialize the DMP class
    my_dmp = Dmp()

    for flight_name, flight_data in data.items():

        for traj_name, trajectory in flight_data.items():

            t = trajectory[:, 0]
            x = trajectory[:, 1:4]

            s = my_dmp.phase(t)
            psi = my_dmp.basis_functions(s)

            w, ftarget = my_dmp.imitate(x,t,s,psi)


            [p,dp,ddp]=my_dmp.generate(w,x[0],x[-1],t,s,psi)

            dmp_data[flight_name][traj_name]["S"] = s
            dmp_data[flight_name][traj_name]["PsV"] = psi
            dmp_data[flight_name][traj_name]["f"] = ftarget
            dmp_data[flight_name][traj_name]["w"] = w
            dmp_data[flight_name][traj_name]["p"] = p

    print("s:", s.shape)
    print("PsV:", psi.shape)

    # Plot the phases
    plots.plot_phases(
        dmp_data,
        num_flights=1
    )

    plots.plot_gaussians(
        dmp_data,
        flight_names=["flight1"],
        traj_names=["traj1"]
    )

    plots.plot_trajectory_comparison(
        data,
        dmp_data,
        flight_names=["flight1"],
        traj_names=["traj1"]
    )
    
    plots.show_all()