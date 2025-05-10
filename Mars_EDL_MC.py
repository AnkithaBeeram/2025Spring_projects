import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
import time
from pathlib import Path


R_MARS = 3396.2e3
G = 6.674e-11      
M_MARS = 6.4171e23 
g0 = G * M_MARS / (R_MARS**2) 

ideal_params = {
        'entry_gamma_deg': -12.0,
        'beta': 100.0,
        'wind_east': 0.0,
        'wind_north': 0.0,
        'landing_lon': 0.0000,
        'landing_lat': 0,
        'landing_vel': 0
    }

# Mars atmospheric model (simplified exponential)
def mars_atmosphere(altitude):
    rho0 = 0.020    
    H = 11.1e3      # Scale height [m]
    alt_clipped = np.maximum(altitude, 0)
    rho = rho0 * np.exp(-alt_clipped / H)
    return rho

def mars_edl_dynamics(state, t, beta, wind_east, wind_north):
    r, theta, phi, v, gamma, psi = state
    altitude = r - R_MARS
    rho = mars_atmosphere(altitude)
    g = G * M_MARS / (r**2)
    v_wind_theta = wind_east / (r * np.cos(phi))  
    v_wind_phi = wind_north / r                  
    v_rel = v 
    drag_acc = 0.5 * rho * v_rel**2 / beta
    dr_dt = v * np.sin(gamma) 
    dtheta_dt = v * np.cos(gamma) * np.sin(psi) / (r * np.cos(phi)) + v_wind_theta
    dphi_dt = v * np.cos(gamma) * np.cos(psi) / r + v_wind_phi
    dv_dt = -drag_acc - g * np.sin(gamma)
    dgamma_dt = (v / r - g / v) * np.cos(gamma)
    dpsi_dt = -v * np.cos(gamma) * np.sin(psi) * np.tan(phi) / r

    return np.array([dr_dt, dtheta_dt, dphi_dt, dv_dt, dgamma_dt, dpsi_dt])

def rk4_step(func, state, t, dt, *args):

    k1 = dt * func(state, t, *args)
    k2 = dt * func(state + 0.5 * k1, t + 0.5 * dt, *args)
    k3 = dt * func(state + 0.5 * k2, t + 0.5 * dt, *args)
    k4 = dt * func(state + k3, t + dt, *args)

    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate_edl(entry_altitude, entry_velocity, entry_gamma, entry_lat, entry_lon, beta, wind_east, wind_north, dt=1.0, max_steps=10000):

    r0 = R_MARS + entry_altitude
    theta0 = entry_lon
    phi0 = entry_lat
    v0 = entry_velocity
    gamma0 = entry_gamma
    psi0 = 0.0 

    state = np.array([r0, theta0, phi0, v0, gamma0, psi0])

    time_points = [0.0]
    altitude = [entry_altitude]
    longitude = [np.degrees(theta0)]
    latitude = [np.degrees(phi0)]
    velocity = [v0]
    gamma = [np.degrees(gamma0)]
    psi = [np.degrees(psi0)]

    t = 0.0
    for _ in range(max_steps):
        state = rk4_step(mars_edl_dynamics, state, t, dt, beta, wind_east, wind_north)
        t += dt
        r, theta, phi, v, gamma_rad, psi_rad = state
        time_points.append(t)
        altitude.append(r - R_MARS)
        longitude.append(np.degrees(theta))
        latitude.append(np.degrees(phi))
        velocity.append(v)
        gamma.append(np.degrees(gamma_rad))
        psi.append(np.degrees(psi_rad))
        if r <= R_MARS:
            break
    trajectory = {
        'time': np.array(time_points),
        'altitude': np.array(altitude),
        'longitude': np.array(longitude),
        'latitude': np.array(latitude),
        'velocity': np.array(velocity),
        'gamma': np.array(gamma),
        'psi': np.array(psi)
    }

    return trajectory

      

def plot_trajectory(trajectory, filename = "Nominal Trajectory"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Altitude vs time
    axs[0, 0].plot(trajectory['time'], trajectory['altitude'] / 10000)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Altitude (km)')
    axs[0, 0].set_title('Altitude vs Time')
    axs[0, 0].grid(True)

    # Velocity vs time
    axs[0, 1].plot(trajectory['time'], trajectory['velocity'])
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Velocity (m/s)')
    axs[0, 1].set_title('Velocity vs Time')
    axs[0, 1].grid(True)

    # Latitude vs longitude
    axs[1, 0].plot(trajectory['longitude'], trajectory['latitude'])
    axs[1, 0].set_xlabel('Longitude (deg)')
    axs[1, 0].set_ylabel('Latitude (deg)')
    axs[1, 0].set_title('Ground Track')
    axs[1, 0].grid(True)

    # Flight path angle vs time
    axs[1, 1].plot(trajectory['time'], trajectory['gamma'])
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Flight Path Angle (deg)')
    axs[1, 1].set_title('Flight Path Angle vs Time')
    axs[1, 1].grid(True)

    plt.tight_layout()
    #plt.show()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")

def simulate_nominal_trajectory():

    # Nominal entry conditions
    entry_altitude = 125000.0  
    entry_velocity = 5500.0    
    entry_gamma_deg = -12.0    
    entry_lat_deg = 0.0        
    entry_lon_deg = 0.0        
    beta = 100.0               
    wind_east = 0.0            
    wind_north = 0.0         

    trajectory = simulate_edl(
        entry_altitude,
        entry_velocity,
        np.radians(entry_gamma_deg),
        np.radians(entry_lat_deg),
        np.radians(entry_lon_deg),
        beta,
        wind_east,
        wind_north
    )

    plot_trajectory(trajectory)

def run_monte_carlo(n_samples):
    
    entry_altitude = 125000.0  
    entry_velocity = 5500.0   
    entry_gamma_deg = -12.0    
    entry_lat_deg = 0.0        
    entry_lon_deg = 0.0        
    beta_nominal = 100.0       
    gamma_std = 0.5           
    beta_rel_std = 0.05      
    wind_std = 15.0     

    
    entry_gamma_samples = np.random.normal(entry_gamma_deg, gamma_std, n_samples)
    beta_samples = np.random.normal(beta_nominal, beta_nominal * beta_rel_std, n_samples)        
    wind_east_samples = np.random.normal(0.0,wind_std, n_samples)
    wind_north_samples = np.random.normal(0.0, wind_std, n_samples)
    
    landing_lon = np.zeros(n_samples)
    landing_lat = np.zeros(n_samples)
    landing_vel = np.zeros(n_samples)
    flight_time = np.zeros(n_samples)

    print(f"Running {n_samples} Monte Carlo simulations...")
    start_time = time.time()

    for i in range(n_samples):
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {i+1}/{n_samples} simulations ({elapsed:.1f} seconds elapsed)")

        entry_gamma = np.radians(entry_gamma_samples[i])
        beta = beta_samples[i]
        wind_east = wind_east_samples[i]
        wind_north = wind_north_samples[i]

        traj = simulate_edl(
            entry_altitude,
            entry_velocity,
            entry_gamma,
            np.radians(entry_lat_deg),
            np.radians(entry_lon_deg),
            beta,
            wind_east,
            wind_north
        )

        landing_lon[i] = traj['longitude'][-1]
        landing_lat[i] = traj['latitude'][-1]
        landing_vel[i] = traj['velocity'][-1]
        flight_time[i] = traj['time'][-1]

    results = pd.DataFrame({
        'entry_gamma_deg': entry_gamma_samples,
        'beta': beta_samples,
        'wind_east': wind_east_samples,
        'wind_north': wind_north_samples,
        'landing_lon': landing_lon,
        'landing_lat': landing_lat,
        'landing_vel': landing_vel,
        'flight_time': flight_time
    })

    print(f"Monte Carlo simulation completed in {time.time() - start_time:.1f} seconds.")

    return results

def analyze_footprint(results):

    mean_lon = np.mean(results['landing_lon'])
    mean_lat = np.mean(results['landing_lat'])
    std_lon = np.std(results['landing_lon'])
    std_lat = np.std(results['landing_lat'])

    radius_lon = 3 * std_lon
    radius_lat = 3 * std_lat

    km_per_deg_lat = 59.0  
    km_per_deg_lon = 59.0 * np.cos(np.radians(mean_lat))  
    radius_lon_km = radius_lon * km_per_deg_lon
    radius_lat_km = radius_lat * km_per_deg_lat
    footprint_area = np.pi * radius_lon_km * radius_lat_km

    corr_matrix = results[['entry_gamma_deg', 'beta', 'wind_east', 'wind_north',
                          'landing_lon', 'landing_lat']].corr()

    results['dist_from_mean_km'] = np.sqrt(
        ((results['landing_lon'] - mean_lon) * km_per_deg_lon)**2 +
        ((results['landing_lat'] - mean_lat) * km_per_deg_lat)**2
    )
    analysis_results = {
        'mean_lon': mean_lon,
        'mean_lat': mean_lat,
        'std_lon': std_lon,
        'std_lat': std_lat,
        'radius_lon_km': radius_lon_km,
        'radius_lat_km': radius_lat_km,
        'footprint_area': footprint_area,
        'corr_matrix': corr_matrix,
        'results': results
    }

    return analysis_results

def plot_footprint(analysis_results, ideal_params, filename1 = "Mars EDL Landing Footprint.png", filename2="Mars EDL Landing Footprint - Distance from Mean.png"):

    results = analysis_results['results']
    mean_lon = analysis_results['mean_lon']
    mean_lat = analysis_results['mean_lat']
    radius_lon_km = analysis_results['radius_lon_km']
    radius_lat_km = analysis_results['radius_lat_km']
    ideal_lon = ideal_params['landing_lon']
    ideal_lat = ideal_params['landing_lat']
    km_per_deg_lat = 59.0
    km_per_deg_lon = 59.0 * np.cos(np.radians(mean_lat))
    radius_lon_deg = radius_lon_km / km_per_deg_lon
    radius_lat_deg = radius_lat_km / km_per_deg_lat
    results['dist_from_ideal_km'] = np.sqrt(
        ((results['landing_lon'] - ideal_lon) * km_per_deg_lon)**2 +
        ((results['landing_lat'] - ideal_lat) * km_per_deg_lat)**2
    )
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(results['landing_lon'], results['landing_lat'],
                   c=results['entry_gamma_deg'], cmap='viridis',
                   alpha=0.6, edgecolors='none', s=30)
    plt.scatter([mean_lon], [mean_lat], color='red', s=100,
              marker='x', label='Mean Landing Site')
    plt.scatter([ideal_lon], [ideal_lat], color='green', s=150,
              marker='*', label='Ideal Landing Site')
    ellipse = Ellipse(xy=(mean_lon, mean_lat), width=2*radius_lon_deg, height=2*radius_lat_deg,
                    edgecolor='red', fc='none', lw=2, label='3σ Ellipse')
    plt.gca().add_patch(ellipse)
    cbar = plt.colorbar(sc)
    cbar.set_label('Entry Flight Path Angle (deg)')
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Mars EDL Landing Footprint - Relative to Ideal Landing')
    plt.grid(True)
    plt.legend()
    info_text = (
        f"Mean Landing Site: ({mean_lon:.4f}°, {mean_lat:.4f}°)\n"
        f"Ideal Landing Site: ({ideal_lon:.4f}°, {ideal_lat:.4f}°)\n"
        f"3σ Ellipse Semi-axes: {radius_lon_km:.2f} km × {radius_lat_km:.2f} km\n"
        f"Footprint Area (3σ): {analysis_results['footprint_area']:.2f} km²\n"
        f"Mean Distance from Ideal: {np.mean(results['dist_from_ideal_km']):.2f} km"
    )
    plt.annotate(info_text, xy=(0.05, 0.05), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    script_dir = Path(__file__).resolve().parent
    save_path = script_dir / filename1
    plt.savefig(save_path)
    plt.close()
    print(f"Relative footprint plot saved to: {save_path}")

    plt.figure(figsize=(12, 10))
    sc = plt.scatter(results['landing_lon'], results['landing_lat'],
                   c=results['dist_from_ideal_km'], cmap='plasma',
                   alpha=0.6, edgecolors='none', s=30)
    plt.scatter([mean_lon], [mean_lat], color='red', s=100,
              marker='x', label='Mean Landing Site')
    plt.scatter([ideal_lon], [ideal_lat], color='green', s=150,
              marker='*', label='Ideal Landing Site')
    ellipse2 = Ellipse(xy=(mean_lon, mean_lat), width=2*radius_lon_deg, height=2*radius_lat_deg,
                     edgecolor='red', fc='none', lw=2, label='3σ Ellipse')
    plt.gca().add_patch(ellipse2)
    cbar = plt.colorbar(sc)
    cbar.set_label('Distance from Ideal Landing Site (km)')
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Mars EDL Landing Footprint - Distance from Ideal Landing Site')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = script_dir / filename2
    plt.savefig(save_path)
    plt.close()
    print(f"Distance MC2 saved to: {save_path}")


def plot_correlation_heatmap(analysis_results, filename='correlation_heatmap.png' ):
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
                           ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')

    plt.title('Correlation Matrix')
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent
    save_path = script_dir / filename
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def plot_mc_distributions(results, show_correlations=False, figsize=(14, 10),filename="MC Distributions"):

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=figsize)
    if show_correlations:
        grid = plt.GridSpec(4, 3, figure=fig)
    else:
        grid = plt.GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    sns.histplot(results['entry_gamma_deg'], kde=True, color='royalblue', ax=ax1)
    ax1.set_title('Entry Flight Path Angle Distribution')
    ax1.set_xlabel('Entry Angle (deg)')
    
    ax2 = fig.add_subplot(grid[0, 1])
    sns.histplot(results['beta'], kde=True, color='forestgreen', ax=ax2)
    ax2.set_title('Ballistic Coefficient Distribution')
    ax2.set_xlabel('Beta (kg/m²)')
    
    ax3 = fig.add_subplot(grid[1, 0])
    sns.histplot(results['wind_east'], kde=True, color='darkorange', ax=ax3)
    ax3.set_title('East Wind Distribution')
    ax3.set_xlabel('Wind Speed (m/s)')
    
    ax4 = fig.add_subplot(grid[1, 1])
    sns.histplot(results['wind_north'], kde=True, color='purple', ax=ax4)
    ax4.set_title('North Wind Distribution')
    ax4.set_xlabel('Wind Speed (m/s)')

    ax5 = fig.add_subplot(grid[2, 0])
    sns.histplot(results['landing_lon'] * 180/np.pi, kde=True, color='crimson', ax=ax5)
    ax5.set_title('Landing Longitude Distribution')
    ax5.set_xlabel('Longitude (deg)')
    
    ax6 = fig.add_subplot(grid[2, 1])
    sns.histplot(results['landing_lat'] * 180/np.pi, kde=True, color='teal', ax=ax6)
    ax6.set_title('Landing Latitude Distribution')
    ax6.set_xlabel('Latitude (deg)')
    
    ax7 = fig.add_subplot(grid[3, 0])
    sns.histplot(results['landing_vel'], kde=True, color='darkred', ax=ax7)
    ax7.set_title('Landing Velocity Distribution')
    ax7.set_xlabel('Velocity (m/s)')
    
    ax8 = fig.add_subplot(grid[3, 1])
    sns.histplot(results['flight_time'], kde=True, color='navy', ax=ax8)
    ax8.set_title('Flight Time Distribution')
    ax8.set_xlabel('Time (s)')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")

def print_analysis_summary(analysis_results):
    print("\n======= Mars EDL Monte Carlo Analysis Summary =======")
    print(f"Mean Landing Site: ({analysis_results['mean_lon']:.4f}°, {analysis_results['mean_lat']:.4f}°)")
    print(f"Standard Deviation: {analysis_results['std_lon']:.4f}° longitude, {analysis_results['std_lat']:.4f}° latitude")
    print(f"3-sigma Ellipse Semi-axes: {analysis_results['radius_lon_km']:.2f} km × {analysis_results['radius_lat_km']:.2f} km")
    print(f"Footprint Area (3-sigma ellipse): {analysis_results['footprint_area']:.2f} km²")

if __name__ == "__main__":
    print("Mars Entry, Descent, and Landing Uncertainty Analysis")
    print("=====================================================")

    print("\nSimulating nominal trajectory...")
    nominal_traj = simulate_nominal_trajectory()
        
    # Get number of samples
    while True:
        try:
            n_samples = int(input("\nEnter number of Monte Carlo samples (10-1000): "))
            if 10 <= n_samples <= 1000:
                break
            else:
                print("Please enter a value between 10 and 1000.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\nRunning Monte Carlo simulation...")
    mc_results = run_monte_carlo(
        n_samples=n_samples
    )

    print("\nPlotting distributions of input and output variables from Monte Carlo simulations...")
    plot_mc_distributions(mc_results, filename="MC_Distributions.png")

    print("\nAnalyzing landing footprint...")
    analysis_results = analyze_footprint(mc_results)

    print("\nPlotting landing footprint relative to ideal landing location...")
    plot_footprint(analysis_results, ideal_params)

    print("\nPlotting correlation heatmap...")
    plot_correlation_heatmap(analysis_results)

    print_analysis_summary(analysis_results)

    mean_dist_from_ideal = np.mean(analysis_results['results']['dist_from_ideal_km'])
    std_dist_from_ideal = np.std(analysis_results['results']['dist_from_ideal_km'])
    
    print("\n===== Distance from Ideal Landing Site =====")
    print(f"Mean distance: {mean_dist_from_ideal:.2f} km")
    print(f"Standard deviation: {std_dist_from_ideal:.2f} km")
    print(f"3-sigma radius: {3*std_dist_from_ideal:.2f} km")

    print("\nAnalysis complete!")
