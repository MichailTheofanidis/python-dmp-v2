from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_data(t, p, title='Position', figure=None, axes=None, save=False, directory=''):

    labels = ["x", "y", "z"]

    # Create figure and axes if they don't already exist
    if figure is None:
        figure, axes = plt.subplots(3, 1, sharex=True)

    for i in range(p.shape[1]):

        axes[i].plot( t, p[:, i], label=title )

        axes[i].set_ylabel(labels[i])
        axes[i].legend()

    axes[-1].set_xlabel("time")

    if save:
        save_fig(figure, title, directory)

    return figure, axes

def plot_flight_trajectories( data, num_flights=2, save=False, directory='' ):

    for flight_idx, (flight_name, flight_data) in enumerate(data.items()):

        # Only plot the requested number of flights
        if flight_idx >= num_flights:
            break

        # Create ONE figure and ONE set of axes for this flight
        figure, axes = plt.subplots(3, 1, sharex=True)

        figure.suptitle(f"Dataset {flight_idx+1}")

        # Plot all trajectories belonging to this flight
        for traj_name, trajectory in flight_data.items():

            t = trajectory[:, 0]
            positions = trajectory[:, 1:4]

            traj_id = traj_name.replace("traj", "")

            figure, axes = plot_trajectory_data( t, positions, title=f"Trajectory {traj_id}", figure=figure, axes=axes, save=False )

        # Improve spacing
        figure.tight_layout()

        # Save once per flight
        if save:
            save_fig( figure,f"{flight_name}", directory)

def plot_3d_flight_trajectories(data, num_flights=None, save=False, directory=''):
        """
        Plot all 3D trajectories for each flight.

        Each flight gets its own figure.
        """

        for flight_idx, (flight_name, flight_data) in enumerate(data.items()):

            # Limit number of flights if requested
            if num_flights is not None and flight_idx >= num_flights:
                break

            # Create one figure for this flight
            figure = plt.figure()
            ax = figure.add_subplot(111, projection='3d')

            figure.suptitle(f"Dataset {flight_idx+1}")

            # Plot all trajectories belonging to this flight
            for traj_name, trajectory in flight_data.items():
                positions = trajectory[:, 1:4]

                traj_id = traj_name.replace("traj", "")

                ax.plot(
                    positions[:, 0],  # x
                    positions[:, 1],  # y
                    positions[:, 2],  # z
                    label=f"Trajectory {traj_id}"
                )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.legend()

            if save:
                save_fig(figure,f"Flight_{flight_name}_3D",directory)

def plot_phases(dmp_data, num_flights=None, title="Phase"):

    figures = []

    flights = list(dmp_data.items())

    if num_flights is not None:
        flights = flights[:num_flights]

    for flight_name, flight_data in flights:

        figure = plt.figure()

        for traj_name, trajectory_data in flight_data.items():

            s = trajectory_data["S"]

            if s is not None:
                plt.plot(s, label=traj_name)

        figure.suptitle(f"{title} - {flight_name}")
        plt.xlabel("Sample")
        plt.ylabel("s")
        plt.legend()
        plt.grid(True)

        figures.append(figure)

    return figures

def plot_gaussians( dmp_data, num_flights=None, flight_names=None, traj_names=None, gaussian_ids=None, title="Gaussians" ):

    figures = []
    labels = ["X", "Y", "Z"]

    # Select flights
    if flight_names is not None:
        flights = [
            (name, dmp_data[name])
            for name in flight_names
            if name in dmp_data
        ]
    else:
        flights = list(dmp_data.items())

        if num_flights is not None:
            flights = flights[:num_flights]

    for flight_name, flight_data in flights:

        for traj_name, trajectory_data in flight_data.items():

            # Select trajectories
            if traj_names is not None and traj_name not in traj_names:
                continue

            s = trajectory_data["S"]
            psv = trajectory_data["PsV"]
            w = trajectory_data["w"]

            if s is None or psv is None:
                continue

            s = np.asarray(s).flatten()
            psv = np.asarray(psv)

            print(
                f"{flight_name} - {traj_name}: "
                f"s={s.shape}, psv={psv.shape}"
            )

            # If weights are not provided
            if w is None:
                w = np.ones((psv.shape[0], 1))
            else:
                w = np.asarray(w)

            # Check dimensions
            if psv.ndim != 2:
                raise ValueError(
                    f"PSV must be 2D, but got shape {psv.shape}"
                )

            if psv.shape[1] != len(s):
                raise ValueError(
                    f"PSV shape {psv.shape} is incompatible with "
                    f"s shape {s.shape}. "
                    f"Expected psv.shape[1] == len(s)."
                )

            # Select Gaussians
            if gaussian_ids is None:
                selected_gaussians = range(psv.shape[0])
            else:
                selected_gaussians = gaussian_ids

                # Check Gaussian indices
                for j in selected_gaussians:
                    if j < 0 or j >= psv.shape[0]:
                        raise ValueError(
                            f"Gaussian index {j} is out of range. "
                            f"Valid range: 0-{psv.shape[0] - 1}"
                        )

            figure = plt.figure()

            figure.suptitle(
                f"{title} - {flight_name} - {traj_name}"
            )

            # One subplot per output dimension
            for i in range(w.shape[1]):

                plt.subplot(w.shape[1], 1, i + 1)

                for j in selected_gaussians:

                    plt.plot(
                        s,
                        psv[j, :] * w[j, i],
                        label=f"G{j}"
                    )

                plt.ylabel(labels[i])
                plt.grid(True)

                # Only show legend if specific Gaussians were selected
                if gaussian_ids is not None:
                    plt.legend()

            plt.xlabel("Phase")

            figures.append(figure)

    return figures

def plot_trajectory_comparison( data, dmp_data, num_flights=None, flight_names=None, traj_names=None, title="Trajectory Comparison"):

    figures = []

    # Select flights
    if flight_names is not None:
        flights = [
            (name, data[name])
            for name in flight_names
            if name in data
        ]
    else:
        flights = list(data.items())

        if num_flights is not None:
            flights = flights[:num_flights]

    for flight_name, flight_data in flights:

        for traj_name, trajectory in flight_data.items():

            # Select trajectories
            if traj_names is not None and traj_name not in traj_names:
                continue

            # Original trajectory
            t = trajectory[:, 0]
            x_original = trajectory[:, 1:4]

            # DMP trajectory
            x_dmp = dmp_data[flight_name][traj_name]["p"]

            if x_dmp is None:
                continue

            x_dmp = np.asarray(x_dmp)

            # Check dimensions
            if x_original.shape != x_dmp.shape:
                raise ValueError(
                    f"Trajectory shape mismatch for "
                    f"{flight_name} - {traj_name}: "
                    f"original={x_original.shape}, "
                    f"DMP={x_dmp.shape}"
                )

            figure = plt.figure()

            figure.suptitle(
                f"{title} - {flight_name} - {traj_name}"
            )

            labels = ["x", "y", "z"]

            # x, y, z subplots
            for i in range(3):

                plt.subplot(3, 1, i + 1)

                plt.plot(
                    t,
                    x_original[:, i],
                    label="Original"
                )

                plt.plot(
                    t,
                    x_dmp[:, i],
                    "--",
                    label="DMP"
                )

                plt.ylabel(labels[i])
                plt.grid(True)

                if i == 0:
                    plt.legend()

            plt.xlabel("Time")

            figure.tight_layout()

            figures.append(figure)

    return figures

def show_all():
    plt.show()

def save_fig(fig, title, directory):
    fig.savefig(directory + '_' + title.lower().replace(' ', '_') + '.png')
