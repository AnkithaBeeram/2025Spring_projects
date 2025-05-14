## Monte Carlo-Based Risk Assessment of a Mars EDL


This project employs Monte Carlo simulations to systematically evaluate risks during the Entry, Descent, and Landing (EDL) phase of a Mars mission. This project provides a comprehensive framework for simulating EDL trajectories, performing Monte Carlo analyses, and visualizing results.

## Features

- **EDL Physics Simulation**
  - Atmospheric density model for Mars
  - Wind profile modeling with altitude variation
  - Parachute deployment dynamics
  - Powered descent phase
  - 3-DOF trajectory integration using RK4

- **Monte Carlo Analysis**
  - Randomized entry conditions
  - Wind uncertainty modeling
  - Statistical landing footprint analysis
  - Correlation analysis of parameters

- **Visualization Capabilities**
  - Trajectory visualization (altitude, velocity, states)
  - Landing footprint plots with confidence ellipses
  - Correlation heatmaps
  - Safety zone analysis
  - Hypothesis testing comparisons

## Running a Simulation

To run the nominal EDL simulation and Monte Carlo analysis:

    python mars_edl_analysis.py

You will be prompted to enter the number of Monte Carlo samples (default: 500). The script will generate trajectory plots, landing footprint visualizations, correlation heatmaps, safety zone diagrams, and a final analysis summary.

## Hypothesis

If the parachute deployment altitude deviates from the nominal 15,000 meters (15 Kms) to 5000 meters (5 Kms), then both mean velocity and mean deceleration after deployment will change measurably (22% and 15% respectively).

Trajectory when parachute deployment is at 15 km:

![alt text](https://github.com/AnkithaBeeram/2025Spring_projects/blob/main/Nominal_Trajectory.png)

Trajectory when parachute deployment is at 5 Km:

![alt text](https://github.com/AnkithaBeeram/2025Spring_projects/blob/main/Hypothesis_Trajectory.png)

Hypothesis to Nominal Comparison:

![alt text](https://github.com/AnkithaBeeram/2025Spring_projects/blob/main/Hypothesis_Comparison.png)

## Contributions

Bhavana and Yukta contributed to the physics models, scientific constants, and calculations, with most of their work found in mdl_edl_simulation.py. Ankitha contributed to the overall code structure, implemented the doctests, and developed the visualizations and analysis that helped interpret and present the results, with major contributions in mars_edl_analysis.py and mars_edl_visualization.py.
