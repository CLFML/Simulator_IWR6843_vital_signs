import re
from typing import Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger('radar_simulator')

class ConfigParser:
    """Parser for TI IWR6843 configuration files"""
    
    def __init__(self):
        self.config = {}
        
    def parse_file(self, filename: str) -> Dict[str, Any]:
        """Parse configuration file and return dictionary of parameters"""
        try:
            file_path = Path(filename)
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {filename}")
                
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                # Skip empty lines or comments
                if line and not line.startswith('%'):
                    self._parse_line(line)
                    
            logger.info(f"Successfully parsed config file: {filename}")
            self._log_interpreted_config()
            return self.config
            
        except Exception as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def _parse_line(self, line: str):
        """Parse single line of configuration"""
        tokens = line.split()
        command = tokens[0]
        params = [self._convert_param(p) for p in tokens[1:]]
        
        self.config[command] = params
        logger.debug(f"Parsed command: {command} with parameters: {params}")

    def _convert_param(self, param: str) -> Any:
        """Convert parameter to appropriate type"""
        try:
            return int(param)
        except ValueError:
            try:
                return float(param)
            except ValueError:
                return param

    def _log_interpreted_config(self):
        """Log human-readable interpretation of the configuration"""
        
        interpretations = {
            'dfeDataOutputMode': self._interpret_dfe_mode,
            'channelCfg': self._interpret_channel_config,
            'adcCfg': self._interpret_adc_config,
            'profileCfg': self._interpret_profile_config,
            'chirpCfg': self._interpret_chirp_config,
            'frameCfg': self._interpret_frame_config,
            'vitalSignsCfg': self._interpret_vitalsigns_config,
            'motionDetection': self._interpret_motion_detection
        }
        
        logger.info("Configuration Interpretation:")
        for cmd, params in self.config.items():
            if cmd in interpretations:
                interpretation = interpretations[cmd](params)
                logger.info(f"\n{interpretation}")

    def _interpret_dfe_mode(self, params: List) -> str:
        mode = params[0]
        modes = {
            1: "Frame based chirps",
            2: "Continuous chirping",
            3: "Advanced frame config"
        }
        return f"""DFE Data Output Mode:
        Mode: {modes.get(mode, 'Unknown')} ({mode})"""

    def _interpret_channel_config(self, params: List) -> str:
        rx_mask, tx_mask = params[0:2]
        return f"""Channel Configuration:
        RX Channels: {bin(rx_mask)[2:].zfill(4)} ({rx_mask})
        TX Channels: {bin(tx_mask)[2:].zfill(4)} ({tx_mask})
        Enabled RX: {[i for i in range(4) if rx_mask & (1 << i)]}
        Enabled TX: {[i for i in range(3) if tx_mask & (1 << i)]}"""

    def _interpret_adc_config(self, params: List) -> str:
        adc_bits = ["12-bit", "14-bit", "16-bit"]
        adc_format = ["Real", "Complex1x", "Complex2x"]
        return f"""ADC Configuration:
        ADC Format: {adc_bits[params[0]]}
        Output Format: {adc_format[params[1]]}"""

    def _interpret_profile_config(self, params: List) -> str:
        return f"""Profile Configuration:
        Profile ID: {params[0]}
        Start Frequency: {params[1]} GHz
        Idle Time: {params[2]} μs
        ADC Start Time: {params[3]} μs
        Ramp End Time: {params[4]} μs
        TX Start Power: {params[5]}
        TX Power: {params[8]}
        Frequency Slope: {params[9]} MHz/μs
        ADC Samples: {params[10]}
        Sample Rate: {params[11]} ksps
        RX Gain: {params[13]} dB"""

    def _interpret_chirp_config(self, params: List) -> str:
        return f"""Chirp Configuration:
        Start Index: {params[0]}
        End Index: {params[1]}
        Profile ID: {params[2]}
        Start Freq Variation: {params[3]} kHz
        Freq Slope Variation: {params[4]} MHz/μs
        Idle Time Variation: {params[5]} μs
        ADC Start Time Variation: {params[6]} μs
        TX Enable Mask: {bin(params[7])[2:].zfill(3)}"""

    def _interpret_frame_config(self, params: List) -> str:
        return f"""Frame Configuration:
        Chirp Start Index: {params[0]}
        Chirp End Index: {params[1]}
        Number of Loops: {params[2]}
        Number of Frames: {params[3]} (0=Infinite)
        Frame Periodicity: {params[4]} ms
        Trigger Select: {'Software' if params[5]==1 else 'Hardware'}
        Frame Trigger Delay: {params[6]} ms"""

    def _interpret_vitalsigns_config(self, params: List) -> str:
        return f"""Vital Signs Configuration:
        Breathing Rate Range: {params[0]}-{params[1]} Hz
        Range Start Index: {params[2]}
        Range End Index: {params[3]}
        Guard Length: {params[4]}
        Noise Threshold: {params[5]}
        Alpha Factor: {params[6]}
        Breathing Energy Min: {params[7]}
        Heart Energy Min: {params[8]}"""

    def _interpret_motion_detection(self, params: List) -> str:
        return f"""Motion Detection Configuration:
        Flag: {'Enabled' if params[0] else 'Disabled'}
        Block Size: {params[1]}
        Threshold: {params[2]}"""
    

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse configuration file
    parser = ConfigParser()
    # config = parser.parse_file('profiles/iwr6843_config.cfg')
    # config = parser.parse_file('profiles/vital_signs_60ghz.cfg')
    config = parser.parse_file('profiles/vod_vs_68xx_10fps.cfg')

    # print(config)

    # # Access parsed parameters
    # profile_params = config.get('profileCfg')
    # frame_params = config.get('frameCfg')