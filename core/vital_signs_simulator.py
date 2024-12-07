from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy import signal  # For bandpass filter implementation

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
   sys.path.append(parent_dir)

from config.radar_config import RadarConfig
from config.human_target import HumanTarget, VitalSigns
from core.signal_model import SignalModel
from core.range_dpu import RangeDPU

logger = logging.getLogger('radar_simulator')
# logger.setLevel(logging.DEBUG)

@dataclass
class VitalSignsProcessor:
    """Vital signs detection and processing"""
    radar_config: RadarConfig = field(default_factory=RadarConfig)
    signal_model: SignalModel = field(default_factory=SignalModel)
    window_size: int = 256  # Number of frames for processing window

    # range_dpu: RangeDPU = field(default_factory=RangeDPU)
    range_dpu: RangeDPU = field(init=False)

    # Processing buffers
    phase_buffer: np.ndarray = field(default_factory=lambda: np.array([]))
    breathing_buffer: np.ndarray = field(default_factory=lambda: np.array([]))
    heartbeat_buffer: np.ndarray = field(default_factory=lambda: np.array([]))


    def __post_init__(self):
        """Initialize after instance creation"""
        # Create RangeDPU instance with parameters from radar config
        self.range_dpu = RangeDPU(
            num_adc_samples=self.radar_config.adc.num_samples,
            range_resolution=self.radar_config.range_resolution,
            # Optionally set range bin search area around expected target
            range_bin_start=int((self.signal_model.target.distance - 0.2) / self.radar_config.range_resolution),
            range_bin_end=int((self.signal_model.target.distance + 0.2) / self.radar_config.range_resolution)
        )
        
        # Initialize processing buffers
        self.phase_buffer = np.array([])
        self.breathing_buffer = np.array([])
        self.heartbeat_buffer = np.array([])

    def process_frame(self, adc_samples: np.ndarray) -> dict:
        """Process single frame of radar data"""
        # Process through Range DPU
        complex_samples, range_bin = self.range_dpu.process_chirp(adc_samples)
        
        # Calculate and unwrap phase
        phase = np.unwrap(np.angle(complex_samples))   

        # Update phase buffer
        if len(self.phase_buffer) < self.window_size:
            #  Use mean phase for vital signs processing
            self.phase_buffer = np.append(self.phase_buffer, np.mean(phase))
        else:
            self.phase_buffer = np.roll(self.phase_buffer, -1)
            self.phase_buffer[-1] = np.mean(phase)
            
        if len(self.phase_buffer) < self.window_size:
            return {}

        # Apply bandpass filters for breathing and heartbeat
        breathing_signal = self._bandpass_filter(
            self.phase_buffer,
            low_freq=0.1,  # 6 breaths/min
            high_freq=0.5  # 30 breaths/min
        )
        
        heartbeat_signal = self._bandpass_filter(
            self.phase_buffer,
            low_freq=0.8,  # 48 beats/min
            high_freq=2.0  # 120 beats/min
        )

        # Update signal buffers
        self.breathing_buffer = breathing_signal
        self.heartbeat_buffer = heartbeat_signal
        
        # Calculate rates and signal properties
        results = {
            'breathing_rate': self._estimate_rate(breathing_signal),
            'heartbeat_rate': self._estimate_rate(heartbeat_signal),
            'breathing_snr': self._calculate_snr(breathing_signal),
            'heartbeat_snr': self._calculate_snr(heartbeat_signal),
            'displacement_mm': self._phase_to_displacement(np.mean(phase))
        }
        
        return results

    def _bandpass_filter(self, signal: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        # Calculate frequency points
        frame_rate = 1/self.radar_config.frame.periodicity
        freqs = np.fft.fftfreq(len(signal), 1/frame_rate)
        
        # Create frequency domain filter
        fft = np.fft.fft(signal)
        mask = (abs(freqs) >= low_freq) & (abs(freqs) <= high_freq)
        fft_filtered = fft * mask
        
        return np.real(np.fft.ifft(fft_filtered))

    def _estimate_rate(self, signal: np.ndarray) -> float:
        """Estimate rate from signal using FFT"""
        frame_rate = 1/self.radar_config.frame.periodicity
        freqs = np.fft.fftfreq(len(signal), 1/frame_rate)
        fft = np.abs(np.fft.fft(signal))
        
        # Find peak frequency
        positive_freqs = freqs[freqs > 0]
        positive_fft = fft[freqs > 0]
        peak_idx = np.argmax(positive_fft)
        
        return positive_freqs[peak_idx] * 60  # Convert Hz to beats/min

    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate SNR of signal in dB"""
        signal_power = np.mean(signal**2)
        noise = signal - np.mean(signal)
        noise_power = np.mean(noise**2)
        
        return 10 * np.log10(signal_power/noise_power)

    def _phase_to_displacement(self, phase: float) -> float:
        """Convert phase to displacement in mm"""
        wavelength = 3e8/self.radar_config.rf.start_freq
        return (wavelength * phase)/(4 * np.pi) * 1000  # Convert to mm

    def plot_signals(self) -> None:
        """Plot vital signs signals and spectrum"""
        try:
            import matplotlib.pyplot as plt
            
            if len(self.phase_buffer) < self.window_size:
                logger.warning("Not enough data for plotting")
                return
                
            # Time vector
            t = np.arange(self.window_size) * self.radar_config.frame.periodicity
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot breathing signal
            plt.subplot(2, 1, 1)
            plt.plot(t, self.breathing_buffer, 'b', label='Breathing')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Breathing Signal ({self._estimate_rate(self.breathing_buffer):.1f} breaths/min)')
            plt.grid(True)
            
            # Plot heartbeat signal
            plt.subplot(2, 1, 2)
            plt.plot(t, self.heartbeat_buffer, 'r', label='Heartbeat')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Heartbeat Signal ({self._estimate_rate(self.heartbeat_buffer):.1f} beats/min)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot signals.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create config and target
    from config.config_parser import ConfigParser
    parser = ConfigParser()
    cfg = parser.parse_file('profiles/vod_vs_68xx_10fps.cfg')
    radar_config = RadarConfig.from_parsed_config(cfg)

    target = HumanTarget(
        distance=1.0,
        vital_signs=VitalSigns(
            breathing_rate=0.3,
            heartbeat_rate=1.17
        )
    )

    # Create signal model and vital signs processor
    signal_model = SignalModel(radar_config=radar_config, target=target)
    processor = VitalSignsProcessor(radar_config=radar_config, signal_model=signal_model)

    # Process multiple frames
    logger.info("Processing vital signs data...")
    num_frames = 300

    for _ in range(num_frames):
        i_signal, q_signal = signal_model.generate_received_signal()
        results = processor.process_frame(i_signal, q_signal)
        
        if results:
            logger.info(f"\nFrame Results:")
            logger.info(f"Breathing Rate estimate: {results['breathing_rate']:.1f} breaths/min, actual rate: {target.vital_signs.breathing_rate*60:.1f} breaths/min")
            logger.info(f"Heartbeat Rate estimate: {results['heartbeat_rate']:.1f} beats/min, actual rate: {target.vital_signs.heartbeat_rate*60:.1f} beats/min)")
            logger.info(f"Mean displacement: {results['displacement_mm']:.2f} mm")

    processor.plot_signals()