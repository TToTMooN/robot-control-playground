"""Run a mass–spring–damper simulation with optional visualization.

This script demonstrates how to simulate a simple 1D mass–spring–damper
system using MuJoCo.  A proportional–derivative (PD) controller is
implemented to bring the mass back to the origin.  The script uses
`tyro` to provide a convenient command‑line interface for specifying
simulation parameters and enabling visualization.

Usage:
    python mass_spring.py --duration 5.0 --kp 50.0 --kd 10.0 --initial-pos 0.1 --render true

Dependencies:
    - mujoco
    - numpy
    - tyro (for CLI parsing)
    - matplotlib (optional, for plotting)

If `--render` is passed and your environment supports OpenGL, an
interactive MuJoCo viewer will display the simulation.  If `--plot` is
passed and matplotlib is installed, the script will generate a simple
position/time plot at the end of the run.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional
from pathlib import Path

import numpy as np

import mujoco
from robot_control_playground.simulators.mujoco_sim import MujocoSim

# Try to import the built-in viewer; fall back to mujoco_viewer if available.
try:
    from mujoco import viewer as mj_viewer           # MuJoCo’s C viewer
    HAVE_MUJOCO_VIEWER = True
except Exception:
    try:
        import mujoco_viewer as mj_viewer            # pure-Python fallback
        print("Using mujoco_viewer")
        HAVE_MUJOCO_VIEWER = True
    except Exception:
        HAVE_MUJOCO_VIEWER = False


try:
    import tyro
except ImportError as e:
    raise ImportError(
        "This example requires the 'tyro' package. Install it with pip install tyro.") from e


@dataclasses.dataclass
class MassSpringConfig:
    """Configuration for the mass–spring–damper simulation."""
    duration: float = 5.0  # total simulation time in seconds
    timestep: Optional[float] = None  # use model default if None
    kp: float = 50.0  # proportional gain (spring constant)
    kd: float = 10.0  # derivative gain (damper constant)
    initial_pos: float = 0.1  # initial displacement from equilibrium
    initial_vel: float = 0.0  # initial velocity
    render: bool = False  # whether to launch the interactive viewer
    plot: bool = False  # whether to plot the trajectory after simulation


def run_sim(config: MassSpringConfig) -> None:
    # Load the mass–spring model
    root_dir = Path(__file__).parent.parent
    sim = MujocoSim(str(root_dir / "models" / "mass_spring.xml"))
    if config.timestep is not None:
        sim.model.opt.timestep = config.timestep
    dt = sim.model.opt.timestep

    # Reset the state with initial position and velocity
    qpos = np.zeros(sim.model.nq)
    qvel = np.zeros(sim.model.nv)
    qpos[0] = config.initial_pos
    qvel[0] = config.initial_vel
    sim.reset(qpos, qvel)

    # Logging
    positions = []
    times = []

    # Optional viewer
    viewer = None
    if config.render and HAVE_MUJOCO_VIEWER:
        viewer = mj_viewer.launch(sim.model, sim.data)
    elif config.render and not HAVE_MUJOCO_VIEWER:
        print("No MuJoCo viewer available on this installation; run with "
            "`pip install mujoco_viewer` or omit --render.")

    steps = int(config.duration / dt)
    for _ in range(steps):
        t = sim.data.time
        pos = sim.data.qpos[0]
        vel = sim.data.qvel[0]
        # PD control to drive the mass to zero
        u = -config.kp * pos - config.kd * vel
        # Step simulation
        sim.step([u])
        positions.append(pos)
        times.append(t)
        if viewer is not None:
            viewer.sync()  # update the viewer scene

    if viewer is not None:
        viewer.close()

    # Plot results if requested
    if config.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; cannot plot results.")
        else:
            plt.figure()
            plt.plot(times, positions)
            plt.xlabel("Time [s]")
            plt.ylabel("Position [m]")
            plt.title("Mass–spring–damper response")
            plt.grid(True)
            plt.show()


def main() -> None:
    config = tyro.cli(MassSpringConfig)
    run_sim(config)


if __name__ == "__main__":
    main()