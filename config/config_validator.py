from typing import Dict, Any, List
import logging

logger = logging.getLogger('radar_simulator')

class ConfigValidator:
    """Validator for IWR6843 radar configuration parameters"""

    def __init__(self):
        # Define valid parameter ranges
        self.valid_ranges = {
            'dfeDataOutputMode': {'valid_values': [1, 2, 3]},
            
            'channelCfg': {
                'rx_mask': {'min': 0, 'max': 15},  # 4 RX channels (0b1111)
                'tx_mask': {'min': 0, 'max': 7}    # 3 TX channels (0b111)
            },
            
            'adcCfg': {
                'num_bits': {'valid_values': [0, 1, 2]},  # 12, 14, 16 bits
                'format': {'valid_values': [0, 1, 2]}     # Real, Complex1x, Complex2x
            },
            
            'profileCfg': {
                'start_freq': {'min': 60, 'max': 64},     # GHz
                'idle_time': {'min': 0, 'max': 1000},     # μs
                'adc_start_time': {'min': 0, 'max': 1000},# μs
                'ramp_end_time': {'min': 0, 'max': 1000}, # μs
                'tx_power': {'min': 0, 'max': 100},       # power index
                'freq_slope': {'min': 0, 'max': 200},     # MHz/μs
                'adc_samples': {'min': 64, 'max': 4096},
                'sample_rate': {'min': 0, 'max': 10000},  # ksps
                'rx_gain': {'min': 0, 'max': 48}          # dB
            },
            
            'frameCfg': {
                'chirp_start_idx': {'min': 0, 'max': 512},
                'chirp_end_idx': {'min': 0, 'max': 512},
                'num_loops': {'min': 1, 'max': 255},
                'num_frames': {'min': 0, 'max': 65535},    # 0 = infinite
                'frame_periodicity': {'min': 0, 'max': 1000}  # ms
            }
        }

    def validate_config(self, config: Dict[str, List]) -> bool:
        """
        Validate complete radar configuration
        Returns True if valid, raises ValueError if invalid
        """
        try:
            # Validate each configuration command
            for cmd, params in config.items():
                self._validate_command(cmd, params)
            
            # Validate inter-parameter dependencies
            self._validate_dependencies(config)
            
            logger.info("Configuration validation successful")
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def _validate_command(self, cmd: str, params: List):
        """Validate individual command parameters"""
        
        if cmd not in self.valid_ranges:
            logger.debug(f"No validation rules defined for command: {cmd}")
            return

        if cmd == 'dfeDataOutputMode':
            if params[0] not in self.valid_ranges[cmd]['valid_values']:
                raise ValueError(f"Invalid DFE mode: {params[0]}")

        elif cmd == 'channelCfg':
            rx_mask, tx_mask = params[0:2]
            if not (self.valid_ranges[cmd]['rx_mask']['min'] <= rx_mask <= self.valid_ranges[cmd]['rx_mask']['max']):
                raise ValueError(f"Invalid RX mask: {rx_mask}")
            if not (self.valid_ranges[cmd]['tx_mask']['min'] <= tx_mask <= self.valid_ranges[cmd]['tx_mask']['max']):
                raise ValueError(f"Invalid TX mask: {tx_mask}")

        elif cmd == 'adcCfg':
            if params[0] not in self.valid_ranges[cmd]['num_bits']['valid_values']:
                raise ValueError(f"Invalid ADC bits config: {params[0]}")
            if params[1] not in self.valid_ranges[cmd]['format']['valid_values']:
                raise ValueError(f"Invalid ADC format: {params[1]}")

        elif cmd == 'profileCfg':
            # profileCfg parameter order:
            # [0]: Profile ID
            # [1]: Start frequency (GHz)
            # [2]: Idle time (μs)
            # [3]: ADC start time (μs)
            # [4]: Ramp end time (μs)
            # [5]: TX start power (dB)
            # [6]: TX phase shifter
            # [7]: TX start time (μs)
            # [8]: TX power (dB)
            # [9]: Frequency slope (MHz/μs)
            # [10]: ADC samples
            # [11]: ADC sampling freq (ksps)
            # [12]: HP corner freq 1
            # [13]: HP corner freq 2
            # [14]: RX gain (dB)
            
            ranges = self.valid_ranges[cmd]
            self._validate_range(params[1], ranges['start_freq'], "Start frequency")
            self._validate_range(params[2], ranges['idle_time'], "Idle time")
            self._validate_range(params[3], ranges['adc_start_time'], "ADC start time")
            self._validate_range(params[4], ranges['ramp_end_time'], "Ramp end time")
            self._validate_range(params[8], ranges['tx_power'], "TX power")
            self._validate_range(params[9], ranges['freq_slope'], "Frequency slope")
            self._validate_range(params[10], ranges['adc_samples'], "ADC samples")  # Changed from params[11]
            self._validate_range(params[11], ranges['sample_rate'], "Sample rate")  # Changed from params[12]
            self._validate_range(params[13], ranges['rx_gain'], "RX gain")  # Changed position

        elif cmd == 'frameCfg':
            ranges = self.valid_ranges[cmd]
            self._validate_range(params[0], ranges['chirp_start_idx'], "Chirp start index")
            self._validate_range(params[1], ranges['chirp_end_idx'], "Chirp end index")
            self._validate_range(params[2], ranges['num_loops'], "Number of loops")
            self._validate_range(params[3], ranges['num_frames'], "Number of frames")
            self._validate_range(params[4], ranges['frame_periodicity'], "Frame periodicity")

    def _validate_range(self, value: float, range_dict: Dict, param_name: str):
        """Validate if value is within specified range"""
        if 'valid_values' in range_dict:
            if value not in range_dict['valid_values']:
                raise ValueError(f"{param_name} {value} not in valid values: {range_dict['valid_values']}")
        else:
            if not (range_dict['min'] <= value <= range_dict['max']):
                raise ValueError(f"{param_name} {value} outside valid range [{range_dict['min']}, {range_dict['max']}]")

    def _validate_dependencies(self, config: Dict[str, List]):
        """Validate inter-parameter dependencies"""
        try:
            # Validate profile and frame timing
            if 'profileCfg' in config and 'frameCfg' in config:
                profile = config['profileCfg']
                frame = config['frameCfg']
                
                # Calculate chirp time
                chirp_time = profile[4]  # ramp_end_time
                
                # Calculate frame time
                num_chirps = (frame[1] - frame[0] + 1) * frame[2]  # (end_idx - start_idx + 1) * num_loops
                frame_time = num_chirps * chirp_time
                
                # Validate frame timing (converted to ms)
                frame_time_ms = frame_time / 1000
                if frame[4] <= frame_time_ms:  # frame_periodicity <= frame_time
                    raise ValueError(
                        f"Frame periodicity ({frame[4]} ms) must be greater than "
                        f"total frame time ({frame_time_ms:.3f} ms)")

            # Validate ADC sampling
            if 'profileCfg' in config:
                profile = config['profileCfg']
                
                # Validate ADC sampling time fits within chirp
                sampling_time = (profile[11] / profile[12]) if profile[12] != 0 else 0  # adc_samples / sample_rate
                if sampling_time > (profile[4] - profile[3]):  # ramp_end_time - adc_start_time
                    raise ValueError(
                        f"ADC sampling time ({sampling_time:.3f} μs) exceeds available time "
                        f"({profile[4] - profile[3]} μs)")

            logger.debug("Inter-parameter dependency validation successful")

        except Exception as e:
            logger.error(f"Dependency validation failed: {e}")
            raise ValueError(f"Inter-parameter dependency validation failed: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse and validate configuration file
    from config_parser import ConfigParser
    
    parser = ConfigParser()
    validator = ConfigValidator()
    
    config = parser.parse_file('profiles/vod_vs_68xx_10fps.cfg')
    
    # Debug print
    if 'profileCfg' in config:
        print("\nProfile Config parameters:")
        print(f"Length: {len(config['profileCfg'])}")
        print(f"Values: {config['profileCfg']}")
    
    validator.validate_config(config)