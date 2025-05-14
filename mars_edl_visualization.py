import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

class MarsEDLVisualization:
    """Class for visualizing Mars Entry, Descent, and Landing (EDL) simulation results.
    
    This class provides methods for creating various plots and visualizations of EDL
    trajectories, landing footprints, and Monte Carlo simulation results.
    """
    
    def __init__(self):
        """Initialize the visualization class."""
        pass
    
    def plot_trajectory(self, 
                       trajectory: Dict[str, np.ndarray], 
                       title: str = "EDL Trajectory",
                       filename: Optional[Union[str, Path]] = None) -> None:
        """Create a comprehensive plot of the EDL trajectory.
        
        Args:
            trajectory: Dictionary containing trajectory data with keys:
                - time: Time points [s]
                - altitude: Altitude above Mars surface [m]
                - longitude: Longitude [deg]
                - latitude: Latitude [deg]
                - velocity: Velocity [m/s]
                - gamma: Flight path angle [deg]
                - psi: Heading angle [deg]
                - parachute_state: Parachute deployment state [0-1]
                - powered_descent_state: Powered descent state [0-1]
                - propellant: Propellant mass [kg]
            title: Title for the plot
            filename: If provided, save plot to this file instead of displaying
        
        Returns:
            None
        """
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))

        # Altitude vs velocity
        axs[0, 0].plot(trajectory['velocity'], trajectory['altitude'] / 1000)
        axs[0, 0].set_xlabel('Velocity (m/s)')
        axs[0, 0].set_ylabel('Altitude (km)')
        axs[0, 0].set_title('Altitude vs Time')
        axs[0, 0].grid(True)

        # Altitude vs time
        axs[1, 0].plot(trajectory['time'], trajectory['altitude'] / 1000)
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Altitude (km)')
        axs[1, 0].set_title('Altitude vs Time')
        axs[1, 0].grid(True)

        # Velocity vs time
        axs[0, 1].plot(trajectory['time'], trajectory['velocity'])
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Velocity (m/s)')
        axs[0, 1].set_title('Velocity vs Time')
        axs[0, 1].grid(True)

        # Highlight parachute deployment
        deploy_indices = np.where(np.diff(trajectory['parachute_state'] > 0.5))[0]
        if len(deploy_indices) > 0:
            deploy_idx = deploy_indices[0] + 1
            axs[0, 1].axvline(trajectory['time'][deploy_idx], color='r', linestyle='--',
                         label=f'Parachute deploy at {trajectory["altitude"][deploy_idx]/1000:.1f} km')
            axs[0, 1].legend()
    
        # Flight path angle vs time
        axs[1, 1].plot(trajectory['time'], trajectory['gamma'])
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Flight Path Angle (deg)')
        axs[1, 1].set_title('Flight Path Angle vs Time')
        axs[1, 1].grid(True)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {filename}")
            plt.close()
        else:
            plt.show()

    
    def plot_footprint(self, 
                      analysis_results: Dict[str, Any], 
                      target_lon: float = 0.0, 
                      target_lat: float = 0.0, 
                      filename: Optional[Union[str, Path]] = None) -> None:
        """Plot the landing footprint from Monte Carlo simulation results.
        
        Args:
            analysis_results: Dictionary containing analysis results with keys:
                - results: DataFrame with landing points
                - mean_lon: Mean landing longitude [deg]
                - mean_lat: Mean landing latitude [deg]
                - radius_lon_km: 3-sigma longitude radius [km]
                - radius_lat_km: 3-sigma latitude radius [km]
            target_lon: Target landing longitude [deg]
            target_lat: Target landing latitude [deg]
            filename: If provided, save plot to this file instead of displaying
            
        Returns:
            None
        """
        results = analysis_results['results']
        mean_lon = analysis_results['mean_lon']
        mean_lat = analysis_results['mean_lat']
        radius_lon_km = analysis_results['radius_lon_km']
        radius_lat_km = analysis_results['radius_lat_km']

        km_per_deg_lat = 59.0
        km_per_deg_lon = 59.0 * np.cos(np.radians(mean_lat))
        radius_lon_deg = radius_lon_km / km_per_deg_lon
        radius_lat_deg = radius_lat_km / km_per_deg_lat
        
        plt.figure(figsize=(14, 12))

        sc = plt.scatter(results['landing_lon'], results['landing_lat'],
                       c=results['landing_vel'], cmap='viridis',
                       alpha=0.6, edgecolors='none', s=30)

        plt.scatter([mean_lon], [mean_lat], color='red', s=100,
                  marker='x', label='Mean Landing Site')

        plt.scatter([target_lon], [target_lat], color='green', s=150,
                  marker='*', label='Target Landing Site')

        ellipse = Ellipse(xy=(mean_lon, mean_lat), width=2*radius_lon_deg, height=2*radius_lat_deg,
                        edgecolor='red', fc='none', lw=2, label='3σ Ellipse')
        plt.gca().add_patch(ellipse)

        cbar = plt.colorbar(sc)
        cbar.set_label('Landing Velocity (m/s)')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.title('Mars EDL Landing Footprint')
        plt.grid(True)
        plt.legend()

        info_text = (
            f"Mean Landing Site: ({mean_lon:.4f}°, {mean_lat:.4f}°)\n"
            f"Target Landing Site: ({target_lon:.4f}°, {target_lat:.4f}°)\n"
            f"3σ Ellipse: {radius_lon_km:.2f} km × {radius_lat_km:.2f} km\n"
            f"Footprint Area (3σ): {analysis_results['footprint_area']:.2f} km²\n"
            f"Mean Distance from Target: {analysis_results['mean_distance_from_target']:.2f} km\n"
            f"Mean Landing Velocity: {analysis_results['mean_landing_velocity']:.2f} m/s"
        )
        plt.annotate(info_text, xy=(0.05, 0.05), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Footprint plot saved to: {filename}")
            plt.close()
        else:
            plt.show()

    def plot_correlation_heatmap(self, 
                               analysis_results: Dict[str, Any], 
                               filename: Optional[Union[str, Path]] = None) -> None:
        """Plot a correlation heatmap of Monte Carlo simulation parameters.
        
        Args:
            analysis_results: Dictionary containing analysis results with key:
                - corr_matrix: Correlation matrix DataFrame
            filename: If provided, save plot to this file instead of displaying
            
        Returns:
            None
        """
        corr_matrix = analysis_results['corr_matrix']
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation Coefficient')

        plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
        plt.yticks(np.arange(len(corr_matrix.index)), corr_matrix.index)

        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', 
                               color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to: {filename}")
            plt.close()
        else:
            plt.show()

    def plot_safety_zone(self, 
                        analysis_results: Dict[str, Any], 
                        confidence_levels: List[float] = [0.9, 0.95, 0.99], 
                        filename: Optional[Union[str, Path]] = "Safety_Landing_Zone.png") -> Dict[float, float]:
        """Plot safety landing zones with different confidence levels.
        
        Args:
            analysis_results: Dictionary containing analysis results with keys:
                - results: DataFrame with landing points
                - mean_lon: Mean landing longitude [deg]
                - mean_lat: Mean landing latitude [deg]
            confidence_levels: List of confidence levels to plot (between 0 and 1)
            filename: If provided, save plot to this file instead of displaying
            
        Returns:
            Dict[float, float]: Dictionary mapping confidence levels to radii in km
        """
        results = analysis_results['results']
        mean_lon = analysis_results['mean_lon']
        mean_lat = analysis_results['mean_lat']

        km_per_deg_lat = 59.0
        km_per_deg_lon = 59.0 * np.cos(np.radians(mean_lat))

        distances = np.sqrt(
            ((results['landing_lon'] - mean_lon) * km_per_deg_lon)**2 +
            ((results['landing_lat'] - mean_lat) * km_per_deg_lat)**2
        )

        radii = {conf: np.quantile(distances, conf) for conf in confidence_levels}

        plt.figure(figsize=(12, 10))

        sc = plt.scatter(results['landing_lon'], results['landing_lat'],
                       c=results['landing_vel'], cmap='viridis',
                       alpha=0.6, edgecolors='none', s=30)

        plt.scatter([mean_lon], [mean_lat], color='red', s=100,
                  marker='x', label='Mean Landing Site')
        
        plt.scatter([0], [0], color='green', s=150,
                  marker='*', label='Target Landing Site')

        colors = ['blue', 'purple', 'magenta']
        for i, conf in enumerate(confidence_levels):
            radius_deg_lon = radii[conf] / km_per_deg_lon
            radius_deg_lat = radii[conf] / km_per_deg_lat
            
            ellipse = Ellipse(xy=(mean_lon, mean_lat), 
                            width=2*radius_deg_lon, height=2*radius_deg_lat,
                            edgecolor=colors[i], fc='none', lw=2, 
                            label=f'{conf*100:.0f}% Confidence')
            plt.gca().add_patch(ellipse)
        
        cbar = plt.colorbar(sc)
        cbar.set_label('Landing Velocity (m/s)')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.title('Mars EDL Safety Landing Zones')
        plt.grid(True)
        plt.legend()
        info_text = (
            f"Mean Landing Site: ({mean_lon:.4f}°, {mean_lat:.4f}°)\n"
            f"Target Landing Site: (0.0000°, 0.0000°)\n"
            f"Mean Landing Velocity: {np.mean(results['landing_vel']):.2f} m/s\n" +
            '\n'.join([f"{conf*100:.0f}% zone radius: {radii[conf]:.2f} km" 
                      for conf in confidence_levels])
        )
        plt.annotate(info_text, xy=(0.05, 0.05), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Safety landing zone plot saved to: {filename}")
            plt.close()
        else:
            plt.show()
        
        return radii

    @staticmethod
    def plot_hypothesis_comparison(
                                nominal_landing_vel : float,
                                hypothesis_landing_vel: float,
                                nominal_trajectory: Dict[str, np.ndarray],
                                hypothesis_trajectory: Dict[str, np.ndarray],
                                nominal_results: pd.DataFrame,
                                hypothesis_results: pd.DataFrame,
                                nominal_analysis: Dict[str, Any],
                                hypothesis_analysis: Dict[str, Any]) -> plt.Figure:
        
        """Create a comprehensive comparison plot between nominal and hypothesis EDL scenarios.
        
        This method generates a 2x2 subplot figure comparing various aspects of the nominal
        and hypothesis trajectories, including velocity profiles, landing footprints,
        landing velocity distributions, and landing accuracy."""

        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(nominal_trajectory['altitude']/1000, nominal_trajectory['velocity'], 'b-', 
                label=f'Nominal')
        plt.plot(hypothesis_trajectory['altitude']/1000, hypothesis_trajectory['velocity'], 'r-', 
                label=f'Hypothesis')
        
        nom_deploy_idx = np.where(np.diff(nominal_trajectory['parachute_state'] > 0.5))[0]
        if len(nom_deploy_idx) > 0:
            nom_deploy_idx = nom_deploy_idx[0] + 1
            plt.axvline(nominal_trajectory['altitude'][nom_deploy_idx]/1000, color='b', linestyle='--')
            
        hyp_deploy_idx = np.where(np.diff(hypothesis_trajectory['parachute_state'] > 0.5))[0]
        if len(hyp_deploy_idx) > 0:
            hyp_deploy_idx = hyp_deploy_idx[0] + 1
            plt.axvline(hypothesis_trajectory['altitude'][hyp_deploy_idx]/1000, color='r', linestyle='--')
        
        plt.xlabel('Altitude (km)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity vs. Altitude Comparison')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.scatter(nominal_results['landing_lon'], nominal_results['landing_lat'], 
                   alpha=0.3, color='blue', label='Nominal')
        plt.scatter(hypothesis_results['landing_lon'], hypothesis_results['landing_lat'], 
                   alpha=0.3, color='red', label='Hypothesis')
        
        km_per_deg_lat = 59.0
        km_per_deg_lon = 59.0 * np.cos(np.radians(nominal_analysis['mean_lat']))
        
        nom_ellipse = Ellipse(xy=(nominal_analysis['mean_lon'], nominal_analysis['mean_lat']), 
                            width=2*nominal_analysis['radius_lon_km']/km_per_deg_lon, 
                            height=2*nominal_analysis['radius_lat_km']/km_per_deg_lat,
                            edgecolor='blue', fc='none', lw=2, label='Nominal 3σ')
        plt.gca().add_patch(nom_ellipse)
        
        hyp_ellipse = Ellipse(xy=(hypothesis_analysis['mean_lon'], hypothesis_analysis['mean_lat']), 
                            width=2*hypothesis_analysis['radius_lon_km']/km_per_deg_lon, 
                            height=2*hypothesis_analysis['radius_lat_km']/km_per_deg_lat,
                            edgecolor='red', fc='none', lw=2, label='Hypothesis 3σ')
        plt.gca().add_patch(hyp_ellipse)
        
        plt.scatter([0], [0], color='green', s=100, marker='*', label='Target')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.title('Landing Footprint Comparison')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        def extract_post_parachute_velocities(results, trajectory):
            post_parachute_vels = []
            # Find where parachute deploys in the trajectory
            deploy_indices = np.where(np.diff(trajectory['parachute_state'] > 0.5))[0]
            if len(deploy_indices) > 0:
                deploy_idx = deploy_indices[0] + 1
                post_parachute_vels = trajectory['velocity'][deploy_idx::10]  # Sample every 10th point
            return post_parachute_vels

        nominal_post_vels = extract_post_parachute_velocities(nominal_results, nominal_trajectory)
        hypothesis_post_vels = extract_post_parachute_velocities(hypothesis_results, hypothesis_trajectory)

        plt.hist(nominal_post_vels, bins=30, alpha=0.5, color='blue', label='Nominal')
        plt.hist(hypothesis_post_vels, bins=30, alpha=0.5, color='red', label='Hypothesis')

        plt.axvline(nominal_landing_vel, color='blue', linestyle='--',
               label=f'Nominal Mean: {nominal_landing_vel:.1f} m/s')
        plt.axvline(hypothesis_landing_vel, color='red', linestyle='--',
               label=f'Hypothesis Mean: {hypothesis_landing_vel:.1f} m/s')

        plt.xlabel('Post-Parachute Velocity (m/s)')
        plt.ylabel('Count')
        plt.title('Post-Parachute Velocity Distribution')
        plt.grid(True)
        plt.legend()

        
        plt.subplot(2, 2, 4)
        plt.hist(nominal_results['dist_from_target_km'], bins=30, alpha=0.5, color='blue', label='Nominal')
        plt.hist(hypothesis_results['dist_from_target_km'], bins=30, alpha=0.5, color='red', label='Hypothesis')
        plt.axvline(nominal_analysis['mean_distance_from_target'], color='blue', linestyle='--',
                   label=f'Nominal Mean: {nominal_analysis["mean_distance_from_target"]:.1f} km')
        plt.axvline(hypothesis_analysis['mean_distance_from_target'], color='red', linestyle='--',
                   label=f'Hypothesis Mean: {hypothesis_analysis["mean_distance_from_target"]:.1f} km')
        plt.xlabel('Distance from Target (km)')
        plt.ylabel('Count')
        plt.title('Landing Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle('Parachute Deployment Altitude Comparison', fontsize=16)
        plt.tight_layout()
        
        return plt 