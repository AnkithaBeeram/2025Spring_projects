import numpy as np

from mars_edl_simulation import MarsEDLSimulation
from mars_edl_visualization import MarsEDLVisualization

class MarsEDLAnalysis:
    """Mars Entry, Descent, and Landing (EDL) Analysis.
    
    This class coordinates between simulation and visualization components to perform
    comprehensive analysis of EDL scenarios. It provides methods for running simulations,
    analyzing results, and generating visualizations.
    
    Attributes:
        simulation (MarsEDLSimulation): EDL physics simulation instance
        visualization (MarsEDLVisualization): Visualization tools instance
        output_dir (Path): Directory for saving analysis outputs
    """
    def __init__(self):
        """Initialize the EDL analysis coordinator.
        
        Args:
            output_dir: Directory path for saving analysis outputs.
                       If None, uses current directory.
        """
        self.simulation = MarsEDLSimulation()
        self.visualization = MarsEDLVisualization()

    def print_final_analysis(self, analysis_results, safety_radii):
        """Print comprehensive summary of Monte Carlo EDL simulation analysis results."""

        print("\n== MARS EDL MONTE CARLO ANALYSIS SUMMARY ==")
        
        print("\n----- Landing Accuracy -----")
        print(f"Mean Landing Site: ({analysis_results['mean_lon']:.4f}°, {analysis_results['mean_lat']:.4f}°)")
        print(f"Standard Deviation: {analysis_results['std_lon']:.4f}° longitude, {analysis_results['std_lat']:.4f}° latitude")
        print(f"3-sigma Ellipse Semi-axes: {analysis_results['radius_lon_km']:.2f} km × {analysis_results['radius_lat_km']:.2f} km")
        print(f"Footprint Area (3-sigma ellipse): {analysis_results['footprint_area']:.2f} km²")
        print(f"Mean Distance from Target: {analysis_results['mean_distance_from_target']:.2f} km")
        print(f"Maximum Distance from Target: {analysis_results['max_distance_from_target']:.2f} km")
        
        print("\n----- Landing Safety Zones -----")
        for conf, radius in safety_radii.items():
            print(f"{conf*100:.0f}% Confidence Zone Radius: {radius:.2f} km")
        
        print("\n----- Landing Velocity -----")
        print(f"Mean Landing Velocity: {analysis_results['mean_landing_velocity']:.2f} m/s")
        print(f"Maximum Landing Velocity: {analysis_results['max_landing_velocity']:.2f} m/s")
        
        results = analysis_results['results']
        print("\n----- Critical Parameter Correlations -----")
        corr_matrix = analysis_results['corr_matrix']
        wind_east_lon_corr = corr_matrix.loc['wind_east', 'landing_lon']
        wind_north_lat_corr = corr_matrix.loc['wind_north', 'landing_lat']
        gamma_vel_corr = corr_matrix.loc['entry_gamma_deg', 'landing_vel']
        beta_vel_corr = corr_matrix.loc['beta', 'landing_vel']
        
        print(f"East Wind vs. Landing Longitude: {wind_east_lon_corr:.3f}")
        print(f"North Wind vs. Landing Latitude: {wind_north_lat_corr:.3f}")
        print(f"Entry Flight Path Angle vs. Landing Velocity: {gamma_vel_corr:.3f}")
        print(f"Ballistic Coefficient vs. Landing Velocity: {beta_vel_corr:.3f}")
        
        print("\n----- Risk Assessment -----")
        max_safe_velocity = 55.0  
        high_velocity_landings = sum(results['landing_vel'] > max_safe_velocity)
        high_velocity_percentage = high_velocity_landings / len(results) * 100
        print(f"Landings exceeding {max_safe_velocity} m/s: {high_velocity_percentage:.1f}%")
        
        max_safe_distance = 3.0 
        distant_landings = sum(results['dist_from_target_km'] > max_safe_distance)
        distant_landings_percentage = distant_landings / len(results) * 100
        print(f"Landings more than {max_safe_distance} km from target: {distant_landings_percentage:.1f}%")
        
        print("\n----- Conclusion -----")
        if high_velocity_percentage < 1.0 and distant_landings_percentage < 5.0:
            print("ASSESSMENT: LOW RISK - Nominal EDL parameters appear safe for mission operations.")
        elif high_velocity_percentage < 5.0 and distant_landings_percentage < 10.0:
            print("ASSESSMENT: MODERATE RISK - EDL parameters may need adjustment for improved safety margin.")
        else:
            print("ASSESSMENT: HIGH RISK - EDL parameters should be reconsidered for mission safety.")

    def test_hypothesis(self, n_samples=500):
        """Run simulations to test the hypothesis about parachute deployment altitude."""
        
        # Run nominal case
        print("\nRunning nominal case simulation...")
        nominal_params = self.simulation.NOMINAL_PARAMS.copy()
        nominal_trajectory = self.simulation.simulate_edl(nominal_params)
        self.visualization.plot_trajectory(nominal_trajectory, title="Nominal EDL Trajectory", 
                                        filename="Nominal_Trajectory.png")
        
        print("\nRunning nominal Monte Carlo simulations...")
        nominal_results = self.simulation.run_monte_carlo(n_samples, nominal_params)
        nominal_analysis = self.simulation.analyze_footprint(nominal_results)
        
        print("\nRunning hypothesis case with parachute deployment at 50 km...")
        hypothesis_params = nominal_params.copy()
        hypothesis_params['parachute_deploy_altitude'] = 50000.0
        hypothesis_trajectory = self.simulation.simulate_edl(hypothesis_params)
        self.visualization.plot_trajectory(hypothesis_trajectory, 
                                        title="EDL Trajectory with Parachute at 5 km", 
                                        filename="Hypothesis_Trajectory.png")
        print("\nRunning hypothesis Monte Carlo simulations...")
        hypothesis_results = self.simulation.run_monte_carlo(n_samples, hypothesis_params)
        hypothesis_analysis = self.simulation.analyze_footprint(hypothesis_results)
        
        def get_post_parachute_velocities(results, trajectory):
            deploy_indices = np.where(np.diff(trajectory['parachute_state'] > 0.5))[0]
            if len(deploy_indices) > 0:
                deploy_idx = deploy_indices[0] + 1
                post_parachute_velocities = trajectory['velocity'][deploy_idx:]
                return np.mean(post_parachute_velocities)
            return np.mean(trajectory['velocity'])  

        nominal_landing_vel = get_post_parachute_velocities(nominal_results, nominal_trajectory)
        hypothesis_landing_vel = get_post_parachute_velocities(hypothesis_results, hypothesis_trajectory)
        velocity_increase = (nominal_landing_vel - hypothesis_landing_vel) / nominal_landing_vel * 100

        
        
        plt = self.visualization.plot_hypothesis_comparison(
            nominal_landing_vel, hypothesis_landing_vel, nominal_trajectory, hypothesis_trajectory,
            nominal_results, hypothesis_results,
            nominal_analysis, hypothesis_analysis
        )
        self.visualization.plot_footprint(hypothesis_analysis, filename="Hypothesis_Footprint.png")

        plt.savefig("Hypothesis_Comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n== HYPOTHESIS TEST RESULTS ==")
        print(f"Nominal parachute deployment altitude: {nominal_params['parachute_deploy_altitude']/1000:.1f} km")
        print(f"Hypothesis parachute deployment altitude: {hypothesis_params['parachute_deploy_altitude']/1000:.1f} km")
        print(f"\nNominal mean landing velocity: {nominal_landing_vel:.2f} m/s")
        print(f"Hypothesis mean landing velocity: {hypothesis_landing_vel:.2f} m/s")
        print(f"Velocity increase: {velocity_increase:.1f}%")
        
        
        velocity_hypothesis = abs(velocity_increase) >= 22.0
        
        
        print("\nHypothesis validation:")
        print(f"H1 Part 1 - Velocity increase of 22%: {'VALIDATED' if velocity_hypothesis else 'NOT VALIDATED'}")
        
        return {
            'nominal_results': nominal_results,
            'nominal_analysis': nominal_analysis,
            'hypothesis_results': hypothesis_results,
            'hypothesis_analysis': hypothesis_analysis,
            'velocity_increase': velocity_increase,
            'velocity_hypothesis_validated': velocity_hypothesis,
        }

def main():
    print("Mars Entry, Descent, and Landing Uncertainty Analysis")
    
    analysis = MarsEDLAnalysis()
    
    print("\nSimulating nominal trajectory...")
    nominal_trajectory = analysis.simulation.simulate_edl(analysis.simulation.NOMINAL_PARAMS)
    analysis.visualization.plot_trajectory(nominal_trajectory, title="Nominal EDL Trajectory", 
                                        filename="Nominal_Trajectory.png")
    

    n_samples = 500  # Default value
    user_input = input("\nEnter number of Monte Carlo samples (default 500): ")
    if user_input.strip():
        try:
            n_samples = int(user_input)
            if n_samples < 10:
                print("Using minimum of 10 samples")
                n_samples = 10
            elif n_samples > 5000:
                print("Limiting to maximum of 5000 samples")
                n_samples = 5000
        except ValueError:
            print("Invalid input. Using default of 500 samples.")
    
    print(f"\nRunning Monte Carlo simulation with {n_samples} samples...")
    mc_results = analysis.simulation.run_monte_carlo(n_samples)
    
    print("\nAnalyzing landing footprint...")
    analysis_results = analysis.simulation.analyze_footprint(mc_results)
    
    print("\nGenerating visualization plots...")
    analysis.visualization.plot_footprint(analysis_results, filename="Landing_Footprint.png")
    analysis.visualization.plot_correlation_heatmap(analysis_results, filename="Correlation_Heatmap.png")
    
    print("\nCalculating safety landing zones...")
    safety_radii = analysis.visualization.plot_safety_zone(analysis_results, filename="Safety_Landing_Zone.png")
    
    print("\nTesting hypothesis about parachute deployment altitude...")
    hypothesis_results = analysis.test_hypothesis(n_samples=min(n_samples, 200))
    
    print("\nGenerating final analysis report...")
    analysis.print_final_analysis(analysis_results, safety_radii)
    
    print("\nAll simulations and analysis complete!")

if __name__ == "__main__":
    main() 