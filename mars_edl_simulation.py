import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import time

class MarsEDLSimulation:
    """Mars Entry, Descent, and Landing (EDL) Simulation.
    
    This class provides methods for simulating Mars EDL trajectories and performing
    Monte Carlo analyses of landing scenarios. It includes models for Mars atmosphere,
    wind profiles, parachute deployment, and powered descent.
    
    Attributes:
        R_MARS (float): Mars radius [m]
        G (float): Gravitational constant [m³/kg/s²]
        M_MARS (float): Mars mass [kg]
        g0 (float): Surface gravity [m/s²]
        NOMINAL_PARAMS (Dict[str, float]): Default simulation parameters
    """
    R_MARS = 3396.2e3  # Mars radius [m]
    G = 6.674e-11      # Gravitational constant [m³/kg/s²]
    M_MARS = 6.4171e23 # Mars mass [kg]
    g0 = G * M_MARS / (R_MARS**2)  # Surface gravity [m/s²]

    # Nominal parameters
    NOMINAL_PARAMS = {
        'entry_gamma_deg': -12.0,     # Entry flight path angle [deg]
        'entry_altitude': 80000.0,   # Entry altitude [m]
        'entry_velocity': 5500.0,     # Entry velocity [m/s]
        'entry_lat_deg': 0.0,         # Entry latitude [deg]
        'entry_lon_deg': 0.0,         # Entry longitude [deg]
        'beta': 100.0,                # Ballistic coefficient [kg/m²]
        'parachute_deploy_altitude': 15000.0,  # Nominal parachute deployment altitude [m]
        'parachute_cd_s': 400.0,      # Parachute drag coefficient × area [m²]
        'landing_lon': 0.0,           # Target landing longitude [deg]
        'landing_lat': 0.0,           # Target landing latitude [deg]
        'vehicle_mass': 1000.0,       # Vehicle mass [kg]
        'powered_descent_altitude': 1000.0,  # Start of powered descent [m]
        'powered_descent_thrust': 8000.0,    # Powered descent thrust [N]
        'powered_descent_isp': 225.0,         # Specific impulse [s]
    }

    def __init__(self):
        pass

    @staticmethod
    def mars_atmosphere(altitude):
        """Calculate Mars atmospheric density at a given altitude.
        Args:
            altitude (float or array-like): Altitude above Mars surface in meters  
        Returns:
            float or array-like: Atmospheric density in kg/m³  
        Tests:
            >>> sim = MarsEDLSimulation()
            >>> f"{sim.mars_atmosphere(0):.3f}"  # Surface density
            '0.020'
            >>> f"{sim.mars_atmosphere(11100):.6f}"  # One scale height
            '0.007358'
            >>> f"{sim.mars_atmosphere(-1000):.3f}"  # Below surface (clipped to surface)
            '0.020'
        """
        rho0 = 0.020    # Surface density [kg/m³]
        H = 11.1e3      # Scale height [m]
        alt_clipped = np.maximum(altitude, 0)
        rho = rho0 * np.exp(-alt_clipped / H)
        return rho
        if isinstance(altitude, np.ndarray):
            high_alt_mask = altitude > 50000
            rho[high_alt_mask] = rho0 * np.exp(-50000/H) * np.exp(-(altitude[high_alt_mask]-50000)/(H*0.6))
        elif altitude > 50000:
            rho = rho0 * np.exp(-50000/H) * np.exp(-(altitude-50000)/(H*0.6))
            
        return rho

    @staticmethod
    def mars_wind_profile(altitude, surface_wind_east, surface_wind_north):
        """Calculate wind components at a given altitude based on surface winds.
        Args:
            altitude (float): Altitude above Mars surface in meters
            surface_wind_east (float): Surface wind speed in east direction (m/s)
            surface_wind_north (float): Surface wind speed in north direction (m/s)
        Returns:
            tuple: (wind_east, wind_north) wind components in m/s 
        Tests:
            >>> sim = MarsEDLSimulation()
            >>> e, n = sim.mars_wind_profile(0, 10.0, 5.0)  # Surface winds
            >>> f"{e:.1f}, {n:.1f}"
            '2.0, 1.0'
            >>> e, n = sim.mars_wind_profile(60000, 10.0, 5.0)  # High altitude
            >>> f"{e:.1f}, {n:.1f}"
            '2.0, 1.0'
        """
        norm_alt = np.clip(altitude / 60000.0, 0, 1)
        wind_factor = 3.0 * norm_alt * (1 - norm_alt) + 0.2
        wind_east = surface_wind_east * wind_factor
        wind_north = surface_wind_north * wind_factor
        
        return wind_east, wind_north

    def mars_edl_dynamics(self, state, t, beta, surface_wind_east, surface_wind_north, 
                         parachute_deploy_altitude, parachute_cd_s, vehicle_mass,
                         powered_descent_altitude, powered_descent_thrust, powered_descent_isp):
        r, theta, phi, v, gamma, psi, parachute_deployed, powered_descent_active, propellant_mass = state
        
        altitude = r - self.R_MARS
    
        rho = self.mars_atmosphere(altitude)
        
        g = self.G * self.M_MARS / (r**2)
        wind_east, wind_north = self.mars_wind_profile(altitude, surface_wind_east, surface_wind_north)
        v_wind_theta = wind_east / (r * np.cos(phi))
        v_wind_phi = wind_north / r
    
        if parachute_deployed < 0.5 and altitude <= parachute_deploy_altitude:
            deploy_time = 5.0
            parachute_deployed_rate = 1.0 / deploy_time
        else:
            parachute_deployed_rate = 0.0
        
        if parachute_deployed > 0.5:
            effective_beta = vehicle_mass / (parachute_cd_s * parachute_deployed)
        else:
            effective_beta = beta
        
        v_rel = v
        drag_acc = 0.5 * rho * v_rel**2 / effective_beta
        
        thrust_acc = 0.0
        mdot = 0.0
        
        if altitude <= powered_descent_altitude and propellant_mass > 0 and not powered_descent_active:
            powered_descent_active_rate = 1.0
        else:
            powered_descent_active_rate = 0.0
        
        if powered_descent_active > 0.5 and propellant_mass > 0:
            thrust_acc = powered_descent_thrust / vehicle_mass
            mdot = powered_descent_thrust / (powered_descent_isp * self.g0)
        
        dr_dt = v * np.sin(gamma)
        dtheta_dt = v * np.cos(gamma) * np.sin(psi) / (r * np.cos(phi)) + v_wind_theta
        dphi_dt = v * np.cos(gamma) * np.cos(psi) / r + v_wind_phi
        
        if powered_descent_active > 0.5:
            dv_dt = -drag_acc - g * np.sin(gamma) + thrust_acc * (-np.sin(gamma))
            dgamma_dt = (v / r - g / v) * np.cos(gamma) + thrust_acc * np.cos(gamma) / v
        else:
            dv_dt = -drag_acc - g * np.sin(gamma)
            dgamma_dt = (v / r - g / v) * np.cos(gamma)
        dpsi_dt = -v * np.cos(gamma) * np.sin(psi) * np.tan(phi) / r
        dparachute_deployed_dt = parachute_deployed_rate * (1.0 - parachute_deployed)
        dpowered_descent_active_dt = powered_descent_active_rate * (1.0 - powered_descent_active)
        dpropellant_mass_dt = -mdot
        
        return np.array([dr_dt, dtheta_dt, dphi_dt, dv_dt, dgamma_dt, dpsi_dt, 
                        dparachute_deployed_dt, dpowered_descent_active_dt, dpropellant_mass_dt])

    @staticmethod
    def rk4_step(func, state, t, dt, *args):
        """Perform one step of 4th order Runge-Kutta integration.
        Args:
            func (callable): Function to integrate, must return state derivatives
            state (array-like): Current state vector
            t (float): Current time
            dt (float): Time step
            *args: Additional arguments passed to func
        Returns:
            array-like: New state vector after integration step
        Tests:
            >>> sim = MarsEDLSimulation()
            >>> def simple_oscillator(state, t):
            ...     x, v = state
            ...     return np.array([v, -x])  # Simple harmonic motion
            >>> initial_state = np.array([1.0, 0.0])
            >>> new_state = sim.rk4_step(simple_oscillator, initial_state, 0, 0.1)
            >>> f"{new_state[0]:.6f}, {new_state[1]:.6f}"  # Position and velocity
            '0.995004, -0.099833'
        """
        k1 = dt * func(state, t, *args)
        k2 = dt * func(state + 0.5 * k1, t + 0.5 * dt, *args)
        k3 = dt * func(state + 0.5 * k2, t + 0.5 * dt, *args)
        k4 = dt * func(state + k3, t + dt, *args)
        
        return state + (k1 + 2*k2 + 2*k3 + k4) / 6

    def simulate_edl(self, params, dt=0.1, max_steps=20000):
        """Simulate a complete EDL trajectory with given parameters.
        
        Args:
            params: Dictionary of simulation parameters
            dt: Time step for integration [s]
            max_steps: Maximum number of integration steps
            
        Returns:
            Dictionary containing trajectory data arrays:
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
        """

        entry_altitude = params['entry_altitude']
        entry_velocity = params['entry_velocity']
        entry_gamma = np.radians(params['entry_gamma_deg'])
        entry_lat = np.radians(params['entry_lat_deg'])
        entry_lon = np.radians(params['entry_lon_deg'])
        beta = params['beta']
        wind_east = params.get('wind_east', 0.0)
        wind_north = params.get('wind_north', 0.0)
        parachute_deploy_altitude = params['parachute_deploy_altitude']
        parachute_cd_s = params['parachute_cd_s']
        vehicle_mass = params['vehicle_mass']
        powered_descent_altitude = params['powered_descent_altitude']
        powered_descent_thrust = params['powered_descent_thrust']
        powered_descent_isp = params['powered_descent_isp']
        

        propellant_mass = 0.1 * vehicle_mass
        
        r0 = self.R_MARS + entry_altitude
        theta0 = entry_lon
        phi0 = entry_lat
        v0 = entry_velocity
        gamma0 = entry_gamma
        psi0 = 0.0
        parachute_deployed0 = 0.0
        powered_descent_active0 = 0.0
        
        state = np.array([r0, theta0, phi0, v0, gamma0, psi0, parachute_deployed0, 
                         powered_descent_active0, propellant_mass])
        
        time_points = [0.0]
        altitude = [entry_altitude]
        longitude = [np.degrees(theta0)]
        latitude = [np.degrees(phi0)]
        velocity = [v0]
        gamma = [np.degrees(gamma0)]
        psi = [np.degrees(psi0)]
        parachute_state = [parachute_deployed0]
        powered_descent_state = [powered_descent_active0]
        propellant = [propellant_mass]
        
        
        t = 0.0
        for _ in range(max_steps):
            
            state = self.rk4_step(self.mars_edl_dynamics, state, t, dt, beta, wind_east, wind_north,
                                parachute_deploy_altitude, parachute_cd_s, vehicle_mass,
                                powered_descent_altitude, powered_descent_thrust, powered_descent_isp)
            
            
            t += dt
            r, theta, phi, v, gamma_rad, psi_rad, parachute_deployed, powered_descent_active, propellant_mass = state
            time_points.append(t)
            altitude.append(r - self.R_MARS)
            longitude.append(np.degrees(theta))
            latitude.append(np.degrees(phi))
            velocity.append(v)
            gamma.append(np.degrees(gamma_rad))
            psi.append(np.degrees(psi_rad))
            parachute_state.append(parachute_deployed)
            powered_descent_state.append(powered_descent_active)
            propellant.append(propellant_mass)
    
            if r <= self.R_MARS:
                break
            
            if t > max_steps * dt:
                print("Warning: Maximum simulation time reached")
                break 

        trajectory = {
            'time': np.array(time_points),
            'altitude': np.array(altitude),
            'longitude': np.array(longitude),
            'latitude': np.array(latitude),
            'velocity': np.array(velocity),
            'gamma': np.array(gamma),
            'psi': np.array(psi),
            'parachute_state': np.array(parachute_state),
            'powered_descent_state': np.array(powered_descent_state),
            'propellant': np.array(propellant)
        }
        
        return trajectory

    def run_monte_carlo(self, n_samples, base_params=None, hypothesis=None):

        """Run Monte Carlo simulation with randomized parameters.
        
        Args:
            n_samples: Number of Monte Carlo samples to run
            base_params: Base parameters to modify (uses NOMINAL_PARAMS if None)
            hypothesis: Optional hypothesis to test ('parachute_alt' or None)
            
        Returns:
            DataFrame containing Monte Carlo results with columns:
                - entry_gamma_deg: Entry flight path angles [deg]
                - beta: Ballistic coefficients [kg/m²]
                - wind_east: East wind speeds [m/s]
                - wind_north: North wind speeds [m/s]
                - landing_lon: Landing longitudes [deg]
                - landing_lat: Landing latitudes [deg]
                - landing_vel: Landing velocities [m/s]
                - flight_time: Flight times [s]
                - parachute_deploy_time: Parachute deployment times [s]
                - powered_descent_start_time: Powered descent start times [s]
        """

        if base_params is None:
            base_params = self.NOMINAL_PARAMS.copy()
        
        gamma_std = 0.5            # Standard deviation of entry flight path angle [deg]
        beta_rel_std = 0.05        # Relative standard deviation of ballistic coefficient
        wind_std = 15.0            # Standard deviation of surface winds [m/s]
        
        if hypothesis == 'parachute_alt':
            base_params['parachute_deploy_altitude'] = 5000.0
        
        
        entry_gamma_samples = np.random.normal(base_params['entry_gamma_deg'], gamma_std, n_samples)
        beta_samples = np.random.normal(base_params['beta'], base_params['beta'] * beta_rel_std, n_samples)
        wind_east_samples = np.random.normal(0.0, wind_std, n_samples)
        wind_north_samples = np.random.normal(0.0, wind_std, n_samples)
        
        landing_lon = np.zeros(n_samples)
        landing_lat = np.zeros(n_samples)
        landing_vel = np.zeros(n_samples)
        flight_time = np.zeros(n_samples)
        parachute_deploy_time = np.zeros(n_samples)
        powered_descent_start_time = np.zeros(n_samples)
        
        print(f"Running {n_samples} Monte Carlo simulations...")
        start_time = time.time()
        
        for i in range(n_samples):
            if (i+1) % max(1, n_samples//10) == 0:
                elapsed = time.time() - start_time
                print(f"Completed {i+1}/{n_samples} simulations ({elapsed:.1f} seconds elapsed)")
            
            params = base_params.copy()
            params['entry_gamma_deg'] = entry_gamma_samples[i]
            params['beta'] = beta_samples[i]
            params['wind_east'] = wind_east_samples[i]
            params['wind_north'] = wind_north_samples[i]
            
            traj = self.simulate_edl(params)

            landing_lon[i] = traj['longitude'][-1]
            landing_lat[i] = traj['latitude'][-1]
            landing_vel[i] = traj['velocity'][-1]
            flight_time[i] = traj['time'][-1]
            
            deploy_indices = np.where(np.diff(traj['parachute_state'] > 0.5))[0]
            if len(deploy_indices) > 0:
                parachute_deploy_time[i] = traj['time'][deploy_indices[0] + 1]

            powered_indices = np.where(np.diff(traj['powered_descent_state'] > 0.5))[0]
            if len(powered_indices) > 0:
                powered_descent_start_time[i] = traj['time'][powered_indices[0] + 1]
        
        # Package results
        results = pd.DataFrame({
            'entry_gamma_deg': entry_gamma_samples,
            'beta': beta_samples,
            'wind_east': wind_east_samples,
            'wind_north': wind_north_samples,
            'landing_lon': landing_lon,
            'landing_lat': landing_lat,
            'landing_vel': landing_vel,
            'flight_time': flight_time,
            'parachute_deploy_time': parachute_deploy_time,
            'powered_descent_start_time': powered_descent_start_time
        })
        
        print(f"Monte Carlo simulation completed in {time.time() - start_time:.1f} seconds.")
        
        return results

    def analyze_footprint(self, results, target_lon=0.0, target_lat=0.0):
        """Analyze landing footprint statistics from Monte Carlo results.
    
        Args:
            results (pandas.DataFrame): Monte Carlo simulation results
            target_lon (float): Target landing longitude in degrees
            target_lat (float): Target landing latitude in degrees
        Returns:
            dict: Analysis results including landing statistics
        Tests:
            >>> sim = MarsEDLSimulation()
            >>> # Create a simple test DataFrame with known landing points
            >>> test_data = {
            ...     'landing_lon': [0.1, -0.1, 0.2, -0.2],
            ...     'landing_lat': [0.1, -0.1, 0.2, -0.2],
            ...     'landing_vel': [10.0, 11.0, 12.0, 13.0],
            ...     'entry_gamma_deg': [-12.0] * 4,
            ...     'beta': [100.0] * 4,
            ...     'wind_east': [0.0] * 4,
            ...     'wind_north': [0.0] * 4
            ... }
            >>> results = pd.DataFrame(test_data)
            >>> analysis = sim.analyze_footprint(results)
            >>> f"{analysis['mean_lon']:.1f}"  # Mean longitude
            '0.0'
            >>> f"{analysis['mean_lat']:.1f}"  # Mean latitude
            '0.0'
            >>> f"{analysis['mean_landing_velocity']:.1f}"  # Mean velocity
            '11.5'
        """
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
        results['dist_from_target_km'] = np.sqrt(
            ((results['landing_lon'] - target_lon) * km_per_deg_lon)**2 +
            ((results['landing_lat'] - target_lat) * km_per_deg_lat)**2
        )
        results['dist_from_mean_km'] = np.sqrt(
            ((results['landing_lon'] - mean_lon) * km_per_deg_lon)**2 +
            ((results['landing_lat'] - mean_lat) * km_per_deg_lat)**2
        )
        corr_matrix = results[['entry_gamma_deg', 'beta', 'wind_east', 'wind_north',
                              'landing_lon', 'landing_lat', 'landing_vel']].corr()
        analysis_results = {
            'mean_lon': mean_lon,
            'mean_lat': mean_lat,
            'std_lon': std_lon,
            'std_lat': std_lat,
            'radius_lon_km': radius_lon_km,
            'radius_lat_km': radius_lat_km,
            'footprint_area': footprint_area,
            'mean_distance_from_target': np.mean(results['dist_from_target_km']),
            'max_distance_from_target': np.max(results['dist_from_target_km']),
            'mean_landing_velocity': np.mean(results['landing_vel']),
            'max_landing_velocity': np.max(results['landing_vel']),
            'corr_matrix': corr_matrix,
            'results': results
        }
        
        return analysis_results

if __name__ == "__main__":
    import doctest
    doctest.testmod() 