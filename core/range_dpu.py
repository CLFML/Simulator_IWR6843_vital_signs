from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional
import logging

@dataclass 
class RangeDPU:
    """Hardware Accelerator based Range Processing Unit"""
    num_adc_samples: int          # Number of ADC samples per chirp
    range_resolution: float       # Range resolution in meters
    range_bin_start: int = 0      # Start index for range processing
    range_bin_end: Optional[int] = None   # End index for range processing
    window_type: str = "hanning"  # Window type for range FFT
    window_size: int = 3          # Number of range bins to return around peak
    
    def __post_init__(self):
        """Initialize DPU configurations"""
        self.window = self._get_window()
        self.range_bin_end = self.range_bin_end or self.num_adc_samples
        
        # For debugging/visualization
        self.range_profile = None
        self.selected_bin = None
        
    def _get_window(self) -> np.ndarray:
        """Generate window coefficients"""
        if self.window_type.lower() == "hanning":
            return np.hanning(self.num_adc_samples)
        elif self.window_type.lower() == "hamming":
            return np.hamming(self.num_adc_samples)
        else:
            return np.ones(self.num_adc_samples)

    def process_chirp(self, adc_samples: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process single chirp data
        
        Args:
            adc_samples: Complex ADC samples [num_samples]
            
        Returns:
            complex_samples: Processed complex samples for selected range bin
            range_bin: Index of selected range bin
        """
        # Apply window
        windowed_samples = adc_samples * self.window
        
        # Perform range FFT
        range_fft = np.fft.fft(windowed_samples)
        range_profile = np.abs(range_fft)
        
        # Find range bin with maximum energy within specified range
        search_profile = range_profile[self.range_bin_start:self.range_bin_end]
        max_bin_relative = np.argmax(search_profile)
        max_bin = max_bin_relative + self.range_bin_start
        
        # Extract window of complex samples around peak
        half_window = self.window_size // 2
        start_idx = max(0, max_bin - half_window)
        end_idx = min(len(range_fft), max_bin + half_window + 1)
        complex_samples = range_fft[start_idx:end_idx]
        
        return complex_samples, max_bin
            
        # # Apply window
        # windowed_samples = adc_samples * self.window
        
        # # Perform range FFT
        # range_fft = np.fft.fft(windowed_samples)
        # range_profile = np.abs(range_fft)
        
        # # Store range profile for visualization
        # self.range_profile = range_profile
        
        # # Find range bin with maximum energy within specified range
        # search_profile = range_profile[self.range_bin_start:self.range_bin_end]
        # max_bin_relative = np.argmax(search_profile)
        # max_bin = max_bin_relative + self.range_bin_start
        
        # # Store selected bin info
        # self.selected_bin = max_bin
        
        # return range_fft[max_bin], max_bin

    def get_range_from_bin(self, bin_idx: int) -> float:
        """Convert range bin index to range in meters"""
        return bin_idx * self.range_resolution
        
    def plot_range_profile(self):
        """Plot range profile and selected bin"""
        try:
            import matplotlib.pyplot as plt
            
            if self.range_profile is None:
                logging.warning("No range profile available. Process chirp data first.")
                return
                
            # Create range axis
            range_axis = np.arange(len(self.range_profile)) * self.range_resolution
            
            plt.figure(figsize=(10, 6))
            plt.plot(range_axis, 20*np.log10(np.abs(self.range_profile)))
            
            if self.selected_bin is not None:
                selected_range = self.get_range_from_bin(self.selected_bin)
                plt.axvline(x=selected_range, color='r', linestyle='--', 
                          label=f'Selected bin ({selected_range:.2f}m)')
            
            plt.xlabel('Range (m)')
            plt.ylabel('Magnitude (dB)')
            plt.title('Range Profile')
            plt.grid(True)
            plt.legend()
            plt.show()
            
        except ImportError:
            logging.warning("Matplotlib not installed. Cannot plot range profile.")