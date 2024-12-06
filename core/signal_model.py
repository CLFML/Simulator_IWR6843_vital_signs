from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path
import sys

# Add parent directory to path to import config modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config.radar_config import RadarConfig
from config.human_target import HumanTarget

logger = logging.getLogger('radar_simulator')
# logger.setLevel(logging.DEBUG)

@dataclass
class SignalModel:
    """FMCW Radar Signal Model"""
    radar_config: RadarConfig = field(default_factory=RadarConfig)
    target: HumanTarget = field(default_factory=HumanTarget)
    
    # Internal state
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    chirp_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    noise: np.ndarray = field(default_factory=lambda: np.array([]))

    def generate_chirp_time(self):
        """Generate time vector for a single chirp"""
        num_samples = self.radar_config.adc.num_samples
        sample_time = 1/self.radar_config.adc.sample_rate
        self.t = np.arange(num_samples) * sample_time
        logger.debug(f"Chirp generation:")
        logger.debug(f"Number of samples: {num_samples}")
        logger.debug(f"Sample rate: {self.radar_config.adc.sample_rate} Hz")
        logger.debug(f"Sample time: {sample_time} s")
        logger.debug(f"Chirp duration: {num_samples * sample_time} s")
        return self.t

    def generate_instantaneous_frequency(self, t: np.ndarray) -> np.ndarray:
        """Calculate instantaneous frequency of FMCW chirp"""
        t_adc = t - self.radar_config.rf.adc_start_time
        return self.radar_config.rf.start_freq + self.radar_config.rf.slope * t_adc
        # return self.radar_config.rf.start_freq + self.radar_config.rf.slope * t

    def generate_phase_noise(self, size: int) -> np.ndarray:
        """Generate phase noise"""
        # Simple white gaussian phase noise model
        phase_noise_power = 1e-3  # rad^2
        return np.sqrt(phase_noise_power) * np.random.randn(size)

    def generate_thermal_noise(self, size: int) -> np.ndarray:
        """Generate thermal noise for I/Q channels"""
        kT = 1.38e-23 * 290  # Boltzmann constant * temperature
        noise_power = np.sqrt(kT * 10**(self.radar_config.rf.noise_figure/10))
        return noise_power * (np.random.randn(size) + 1j*np.random.randn(size))

    def calculate_target_displacement(self, t: np.ndarray) -> np.ndarray:
        """Calculate total target displacement including breathing and heartbeat"""
        breathing = self.target.vital_signs.breathing_amplitude * \
                   np.cos(2*np.pi*self.target.vital_signs.breathing_rate*t)
        
        heartbeat = self.target.vital_signs.heartbeat_amplitude * \
                   np.cos(2*np.pi*self.target.vital_signs.heartbeat_rate*t)
        
        if self.target.body_motion.motion_enabled:
            motion = self.target.body_motion.motion_amplitude * np.random.randn(len(t))
        else:
            motion = np.zeros_like(t)
            
        return breathing + heartbeat + motion

    def generate_received_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate received I/Q signal for a single chirp"""
        # Generate time vector if not already done
        if len(self.t) == 0:
            self.generate_chirp_time()
                
        # Calculate target displacement
        displacement = self.calculate_target_displacement(self.t)
        total_distance = self.target.distance + displacement
        logger.debug(f"Target distance range: [{min(total_distance)}, {max(total_distance)}] meters")
        
        # Calculate FMCW parameters
        c = 3e8
        start_freq_hz = self.radar_config.rf.start_freq  
        wavelength = c/start_freq_hz        
        logger.debug(f"Wavelength: {wavelength:.6f} m")
        logger.debug(f"Start frequency: {start_freq_hz/1e9:.2f} GHz")

        logger.debug(f"Frequency slope: {self.radar_config.rf.slope/1e12:.2f} MHz/μs")

        # Calculate instantaneous frequency
        freq_offset = self.radar_config.rf.slope * self.t
        logger.debug(f"Max frequency offset: {max(freq_offset)/1e9:.2f} GHz")

        # Calculate total phase including both distance and chirp terms
        phase_distance = 4*np.pi*total_distance/wavelength
        phase_chirp = 2*np.pi*freq_offset*self.t
        phase = phase_distance + phase_chirp
        
        logger.debug(f"Phase range: [{min(phase)}, {max(phase)}] radians")
        
        # Add phase noise
        phase_noise = self.generate_phase_noise(len(self.t))
        logger.debug(f"Phase noise range: [{min(phase_noise)}, {max(phase_noise)}] radians")
        phase += phase_noise
        
        # Generate I/Q signal 
        signal = np.exp(1j * phase)
        logger.debug(f"Initial signal range: [{np.min(np.abs(signal))}, {np.max(np.abs(signal))}]")
        
        # Add thermal noise
        thermal_noise = self.generate_thermal_noise(len(self.t))
        logger.debug(f"Thermal noise range: [{np.min(np.abs(thermal_noise))}, {np.max(np.abs(thermal_noise))}]")
        signal += thermal_noise

        # Apply gains
        tx_power_linear = 10**(self.radar_config.rf.tx_power/10)  # dBm to linear
        rx_gain_linear = 10**(self.radar_config.rf.rx_gain/20)  # Convert dB to linear scale
        signal *= tx_power_linear * rx_gain_linear

        # # Then apply ADC scaling if needed
        # if self.radar_config.adc.bits:
        #     max_counts = 2**(self.radar_config.adc.bits-1) - 1
        #     signal = np.clip(signal * max_counts, -max_counts, max_counts)
        
        # Apply radar equation attenuation
        path_loss = (wavelength/(4*np.pi*self.target.distance))**4 * \
                self.target.rcs * \
                self.radar_config.antenna.antenna_gain**2
        logger.debug(f"Path loss components:")
        logger.debug(f"- Spreading loss: {(wavelength/(4*np.pi*self.target.distance))**4}")
        logger.debug(f"- Final path loss factor: {path_loss}")
        
        signal *= np.sqrt(path_loss)
        logger.debug(f"Final signal range: [{np.min(np.abs(signal))}, {np.max(np.abs(signal))}]")

        logger.debug(f"Beat frequency expected: {2 * self.radar_config.rf.slope * self.target.distance / 3e8:.2f} Hz")
        logger.debug(f"Phase accumulation rate: {4*np.pi*self.target.distance*self.radar_config.rf.slope/3e8:.2f} Hz")
        # logger.debug(f"Phase accumulation rate: {np.mean(np.diff(phase))/(self.t[1]-self.t[0])/(2*np.pi):.2f} Hz")        

        return np.real(signal), np.imag(signal)

    def get_range_profile(self, i_signal: np.ndarray, q_signal: np.ndarray) -> np.ndarray:
        """Calculate range profile from I/Q signals"""
        # Combine I/Q
        signal = i_signal + 1j*q_signal

        # Debug FFT size and frequency resolution
        Fs = self.radar_config.adc.sample_rate
        N = len(signal)
        N_fft = 8192  # Power of 2, larger than N

        freq_res = Fs/N
        range_res = 3e8 * freq_res/(2*self.radar_config.rf.slope)
        
        # logger.debug(f"FFT size: {N}")
        # logger.debug(f"Freq resolution: {freq_res:.2f} Hz")
        # logger.debug(f"Range resolution: {range_res:.2f} m")        
            
        # Apply window
        window = np.hanning(N)
        padded_signal = np.pad(signal * window, (0, N_fft - N))
        range_fft = np.fft.fft(padded_signal)
        
        range_profile = np.abs(range_fft[:N_fft//2])
        
        return range_profile

    def plot_signals(self) -> None:
        """Plot generated signals"""
        try:
            import matplotlib.pyplot as plt
            
            # Generate signals
            i_signal, q_signal = self.generate_received_signal()
            range_profile = self.get_range_profile(i_signal, q_signal)
            
            # Create figure
            plt.figure() #figsize=(10, 8))
            
            # Plot I/Q signals
            plt.subplot(2, 1, 1)
            plt.plot(self.t*1e6, i_signal, 'b', label='I')
            plt.plot(self.t*1e6, q_signal, 'r', label='Q')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.title('I/Q Signals')
            plt.legend()
            plt.grid(True)
            
            # Plot range profile
            plt.subplot(2, 1, 2)
            range_bins = np.arange(len(range_profile)) * self.radar_config.range_resolution * 2
            plt.plot(range_bins, 20*np.log10(range_profile))
            plt.xlabel('Range (m)')
            plt.ylabel('Magnitude (dB)')
            plt.title('Range Profile')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot signals.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create radar config from a cfg file
    from config.config_parser import ConfigParser
    from config.human_target import VitalSigns, HumanTarget
    
    parser = ConfigParser()
    cfg = parser.parse_file('profiles/vod_vs_68xx_10fps.cfg')
    radar_config = RadarConfig.from_parsed_config(cfg)
    
    # Create human target with custom vital signs
    target = HumanTarget(
        distance=1.0,
        vital_signs=VitalSigns(
            breathing_rate=0.3,    # 18 breaths/min
            heartbeat_rate=1.17    # 70 beats/min
        )
    )
    
    # Create signal model
    signal_model = SignalModel(radar_config=radar_config, target=target)
    
    # Generate and plot signals
    logger.info("Generating radar signals...")
    signal_model.plot_signals()
    
    # Print some signal properties
    i_signal, q_signal = signal_model.generate_received_signal()
    logger.info(f"\nSignal Properties:")
    logger.info(f"Number of samples: {len(i_signal)}")
    logger.info(f"I signal range: [{np.min(i_signal):.2f}, {np.max(i_signal):.2f}]")
    logger.info(f"Q signal range: [{np.min(q_signal):.2f}, {np.max(q_signal):.2f}]")