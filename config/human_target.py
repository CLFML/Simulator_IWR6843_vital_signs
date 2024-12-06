from dataclasses import dataclass, field
import logging

logger = logging.getLogger('radar_simulator')

@dataclass
class VitalSigns:
    """Human vital signs parameters"""
    breathing_rate: float = 0.2      # [Hz] Typical breathing rate 12 breaths/min
    breathing_amplitude: float = 1e-3 # [m] ~1mm chest displacement due to breathing
    
    heartbeat_rate: float = 1.2      # [Hz] Typical heartbeat rate 72 beats/min
    heartbeat_amplitude: float = 1e-4 # [m] ~0.1mm chest displacement due to heartbeat
    
    snr_breathing: float = 20.0      # [dB] Typical SNR for breathing signal
    snr_heartbeat: float = 10.0      # [dB] Typical SNR for heartbeat signal (lower than breathing)

@dataclass 
class BodyMotion:
    """Random body motion parameters"""
    motion_amplitude: float = 5e-5    # [m] Random motion amplitude
    motion_rate: float = 0.1         # [Hz] Rate of random motion
    motion_enabled: bool = True      # Enable/disable random motion

@dataclass
class HumanTarget:
    """Human target parameters for radar simulation"""
    # Position parameters
    distance: float = 1.0            # [m] Distance from radar
    azimuth: float = 0.0            # [deg] Azimuth angle (-60 to +60 degrees typ.)
    elevation: float = 0.0          # [deg] Elevation angle (-60 to +60 degrees typ.)
    
    # Radar cross section parameters
    rcs: float = 0.1                # [m^2] Radar cross section at 60GHz
    rcs_variation: float = 0.02     # [m^2] Variation in RCS due to breathing/motion
    
    # Vital signs and motion - using default_factory
    vital_signs: VitalSigns = field(default_factory=VitalSigns)
    body_motion: BodyMotion = field(default_factory=BodyMotion)

    def validate(self) -> bool:
        """
        Validate human target parameters are within realistic ranges
        Returns True if valid, raises ValueError if invalid
        """
        try:
            # Validate distance
            if not (0.2 <= self.distance <= 5.0):
                raise ValueError(f"Distance {self.distance}m outside valid range [0.2, 5.0]m")
            
            # Validate angles
            if not (-60 <= self.azimuth <= 60):
                raise ValueError(f"Azimuth {self.azimuth}° outside valid range [-60, 60]°")
            if not (-60 <= self.elevation <= 60):
                raise ValueError(f"Elevation {self.elevation}° outside valid range [-60, 60]°")
            
            # Validate vital signs
            if not (0.1 <= self.vital_signs.breathing_rate <= 0.5):  # 6-30 breaths/min
                raise ValueError(f"Breathing rate {self.vital_signs.breathing_rate}Hz outside normal range")
            if not (0.8 <= self.vital_signs.heartbeat_rate <= 3.0):  # 48-180 beats/min
                raise ValueError(f"Heart rate {self.vital_signs.heartbeat_rate}Hz outside normal range")
            
            # Validate motion
            if self.body_motion.motion_amplitude > 1e-3:  # Max 1mm random motion
                raise ValueError(f"Motion amplitude {self.body_motion.motion_amplitude}m too large")
            
            logger.info("Human target parameters validation successful")
            self._log_target_info()
            return True
            
        except ValueError as e:
            logger.error(f"Human target validation failed: {e}")
            raise

    def _log_target_info(self):
        """Log human target parameters in a readable format"""
        logger.info("\nHuman Target Configuration:")
        logger.info(f"Position: {self.distance:.2f}m, Az: {self.azimuth:.1f}°, El: {self.elevation:.1f}°")
        logger.info(f"RCS: {self.rcs:.3f}m² ± {self.rcs_variation:.3f}m²")
        logger.info("\nVital Signs:")
        logger.info(f"Breathing: {self.vital_signs.breathing_rate*60:.1f} breaths/min, "
                   f"amplitude: {self.vital_signs.breathing_amplitude*1000:.2f}mm")
        logger.info(f"Heartbeat: {self.vital_signs.heartbeat_rate*60:.1f} beats/min, "
                   f"amplitude: {self.vital_signs.heartbeat_amplitude*1000:.2f}mm")
        logger.info(f"SNR - Breathing: {self.vital_signs.snr_breathing:.1f}dB, "
                   f"Heartbeat: {self.vital_signs.snr_heartbeat:.1f}dB")
        if self.body_motion.motion_enabled:
            logger.info("\nRandom Motion:")
            logger.info(f"Amplitude: {self.body_motion.motion_amplitude*1000:.2f}mm, "
                       f"Rate: {self.body_motion.motion_rate:.2f}Hz")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and validate a human target
    target = HumanTarget(
        distance=1.5,
        azimuth=30,
        elevation=0,
        vital_signs=VitalSigns(
            breathing_rate=0.3,    # 18 breaths/min
            heartbeat_rate=1.17    # 70 beats/min
        )
    )
    
    target.validate()