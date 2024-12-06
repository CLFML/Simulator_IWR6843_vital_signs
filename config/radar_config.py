from dataclasses import dataclass, field
import numpy as np
import logging

@dataclass
class AntennaConfig:
    """Antenna configuration parameters"""
    num_rx: int = 4
    num_tx: int = 3
    rx_channels: int = 0b1111  # RX channel mask
    tx_channels: int = 0b101   # TX channel mask
    azimuth_fov: float = 120   # [deg]
    elevation_fov: float = 120  # [deg]
    antenna_gain: float = 8     # [dBi]

@dataclass
class RFConfig:
    """RF front-end configuration"""
    start_freq: float = 60.25e9  # [Hz]
    slope: float = 100e12       # [Hz/s]
    idle_time: float = 7e-6     # [s]
    adc_start_time: float = 6e-6 # [s]
    ramp_end_time: float = 60e-6 # [s]
    tx_power: float = 12         # [dBm]
    noise_figure: float = 14     # [dB]
    rx_gain: float = 30          # [dB]

@dataclass
class ADCConfig:
    """ADC configuration"""
    num_samples: int = 256
    sample_rate: float = 5e6   # [Hz]
    format: str = "COMPLEX"    # "REAL" or "COMPLEX"
    bits: int = 16            # ADC resolution

@dataclass
class ChirpConfig:
    """Single chirp configuration"""
    profile_id: int = 0
    start_idx: int = 0
    end_idx: int = 0
    tx_mask: int = 1          # TX antenna enable mask
    
@dataclass
class FrameConfig:
    """Frame timing configuration"""
    chirp_loops: int = 64     # Number of chirps per frame
    periodicity: float = 100e-3  # [s] Frame period
    trigger_mode: str = "SOFTWARE"  # "SOFTWARE" or "HARDWARE"

@dataclass
class RadarConfig:
    """Complete radar configuration"""
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    adc: ADCConfig = field(default_factory=ADCConfig)
    chirp: ChirpConfig = field(default_factory=ChirpConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)    

    @classmethod
    def from_parsed_config(cls, parsed_config: dict):
        """Create RadarConfig from parsed configuration dictionary"""
        # Extract antenna config
        antenna = AntennaConfig(
            rx_channels=parsed_config['channelCfg'][0],
            tx_channels=parsed_config['channelCfg'][1]
        )

        # Extract RF config
        profile = parsed_config['profileCfg']
        rf = RFConfig(
            start_freq=profile[1] * 1e9,  # Convert GHz to Hz
            slope=profile[9] * 1e12,      # Convert MHz/μs to Hz/s
            idle_time=profile[2] * 1e-6,  # Convert μs to s
            adc_start_time=profile[3] * 1e-6,
            ramp_end_time=profile[4] * 1e-6,
            tx_power=profile[8],
            noise_figure=profile[7],
            rx_gain=profile[13]            
        )

        # Extract ADC config
        # Calculate ADC sampling rate from number of samples and ramp time
        sampling_time = (profile[4] - profile[3]) * 1e-6  # ramp_end_time - adc_start_time in seconds
        sample_rate = profile[10] / sampling_time  # samples / time = Hz

        adc = ADCConfig(
            num_samples=profile[10],      # ADC samples
            sample_rate=profile[11] * 1e3 if profile[11] != 0 else (profile[10] / ((profile[4] - profile[3]) * 1e-6)),  # Calculate from samples and time if not specified
            format="COMPLEX" if parsed_config['adcCfg'][1] == 1 else "REAL",
            bits=16 if parsed_config['adcCfg'][0] == 2 else (14 if parsed_config['adcCfg'][0] == 1 else 12)
        )

        # Extract frame config
        frame_cfg = parsed_config['frameCfg']
        frame = FrameConfig(
            chirp_loops=frame_cfg[2],
            periodicity=frame_cfg[4] * 1e-3,  # Convert ms to s
            trigger_mode="SOFTWARE" if frame_cfg[5] == 1 else "HARDWARE"
        )

        return cls(
            antenna=antenna,
            rf=rf,
            adc=adc,
            frame=frame
        )    

    @property
    def bandwidth(self) -> float:
        """Calculate chirp bandwidth in Hz"""
        return self.rf.slope * (self.rf.ramp_end_time - self.rf.adc_start_time)

    @property
    def range_resolution(self) -> float:
        """Calculate range resolution in meters"""
        c = 3e8  # speed of light
        return c / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """Calculate maximum unambiguous range in meters"""
        c = 3e8  # speed of light
        return (self.adc.sample_rate * c) / (2 * self.rf.slope)

    @property
    def velocity_resolution(self) -> float:
        """Calculate velocity resolution in m/s"""
        wavelength = 3e8 / self.rf.start_freq
        chirp_time = self.rf.ramp_end_time + self.rf.idle_time
        return wavelength / (2 * self.frame.chirp_loops * chirp_time)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('radar_simulator')
    
    # # Method 1: Create radar config with default values
    # radar_default = RadarConfig()
    # logger.info("\nDefault Radar Configuration:")
    # logger.info(f"Range Resolution: {radar_default.range_resolution*1000:.2f} mm")
    # logger.info(f"Maximum Range: {radar_default.max_range:.2f} m")
    # logger.info(f"Velocity Resolution: {radar_default.velocity_resolution:.2f} m/s")
    
    # Method 2: Create radar config from parsed .cfg file
    from config_parser import ConfigParser
    
    parser = ConfigParser()
    cfg = parser.parse_file('profiles/vod_vs_68xx_10fps.cfg')
    print(cfg)
    
    radar_from_cfg = RadarConfig.from_parsed_config(cfg)
    logger.info("\nRadar Configuration from .cfg file:")
    logger.info(f"Start Frequency: {radar_from_cfg.rf.start_freq/1e9:.2f} GHz")
    logger.info(f"Bandwidth: {radar_from_cfg.bandwidth/1e6:.2f} MHz")
    logger.info(f"Range Resolution: {radar_from_cfg.range_resolution*1000:.2f} mm")
    logger.info(f"Maximum Range: {radar_from_cfg.max_range:.2f} m")
    logger.info(f"Frame Period: {radar_from_cfg.frame.periodicity*1000:.1f} ms")
    logger.info(f"ADC Samples: {radar_from_cfg.adc.num_samples}")
    logger.info(f"TX Channels: {bin(radar_from_cfg.antenna.tx_channels)}")
    logger.info(f"RX Channels: {bin(radar_from_cfg.antenna.rx_channels)}")
    logger.info(f"Trigger Mode: {radar_from_cfg.frame.trigger_mode}")
    logger.info(f"ADC Format: {radar_from_cfg.adc.format}")
    logger.info(f"ADC Bits: {radar_from_cfg.adc.bits}")
    logger.info(f"ADC Sample Rate: {radar_from_cfg.adc.sample_rate/1e3:.2f} ksps")
    logger.info(f"RX Gain: {radar_from_cfg.rf.rx_gain} dB")
    # logger.info(f"Antenna Gain: {radar_from_cfg.antenna.antenna_gain} dBi")
    # logger.info(f"TX Power: {radar_from_cfg.rf.tx_power} dBm")
    # logger.info(f"Noise Figure: {radar_from_cfg.rf.noise_figure} dB")
    # logger.info(f"Chirp Loops: {radar_from_cfg.frame.chirp_loops}")
    # logger.info(f"Velocity Resolution: {radar_from_cfg.velocity_resolution:.2f} m/s")
    # logger.info(f"Chirp Profile ID: {radar_from_cfg.chirp.profile_id}")
    # logger.info(f"TX Mask: {bin(radar_from_cfg.chirp.tx_mask)}")
    # logger.info(f"Idle Time: {radar_from_cfg.rf.idle_time*1e6:.2f} μs")
    # logger.info(f"ADC Start Time: {radar_from_cfg.rf.adc_start_time*1e6:.2f} μs")
    # logger.info(f"Ramp End Time: {radar_from_cfg.rf.ramp_end_time*1e6:.2f} μs")
    # logger.info(f"Antenna Azimuth FOV: {radar_from_cfg.antenna.azimuth_fov} deg")
    # logger.info(f"Antenna Elevation FOV: {radar_from_cfg.antenna.elevation_fov} deg")
    # logger.info(f"Frame Trigger Mode: {radar_from_cfg.frame.trigger_mode}")
    # logger.info(f"Frame Periodicity: {radar_from_cfg.frame.periodicity*1000:.2f} ms")
    