from pathlib import Path
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from config.config_parser import ConfigParser
from config.radar_config import RadarConfig
from config.human_target import HumanTarget, VitalSigns
from core.signal_model import SignalModel
from core.vital_signs_simulator import VitalSignsProcessor

class RadarSimulator:
   def __init__(self, config_file: str):
       self.logger = logging.getLogger('radar_simulator')
       
       # Initialize components
       parser = ConfigParser()
       cfg = parser.parse_file(config_file)
       self.radar_config = RadarConfig.from_parsed_config(cfg)
       
       self.target = HumanTarget(
           distance=1.0,
           vital_signs=VitalSigns(
               breathing_rate=0.3,    # 18 breaths/min
               heartbeat_rate=1.17    # 70 beats/min
           )
       )
       
       self.signal_model = SignalModel(
           radar_config=self.radar_config, 
           target=self.target
       )
       
       self.processor = VitalSignsProcessor(
           radar_config=self.radar_config,
           signal_model=self.signal_model
       )

   def run_simulation(self, num_frames: int = 300) -> List[np.ndarray]:
        breathing_rate = np.zeros(num_frames)
        heartbeat_rate = np.zeros(num_frames)
        displacement = np.zeros(num_frames)
         
        for frame in range(num_frames):
            i_signal, q_signal = self.signal_model.generate_received_signal()
           
            results = self.processor.process_frame(i_signal, q_signal)

            if results:
                breathing_rate[frame] = results['breathing_rate']
                heartbeat_rate[frame] = results['heartbeat_rate']
                displacement[frame] = results['displacement_mm']

        return breathing_rate, heartbeat_rate, displacement
  
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()]
        )    

    config_file = Path('profiles/vod_vs_68xx_10fps.cfg')
    simulator = RadarSimulator(str(config_file))
    results = simulator.run_simulation(num_frames=1000)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(results[0], label='Breathing Rate')
    ax[0].plot([simulator.target.vital_signs.breathing_rate*60]*len(results[0]), label='Actual Breathing Rate', linestyle='--', color='r')
    ax[0].set_title('Breathing Rate')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Breaths/min')
    ax[0].grid(True)
    
    ax[1].plot(results[1], label='Heartbeat Rate')
    ax[1].plot([simulator.target.vital_signs.heartbeat_rate*60]*len(results[1]), label='Actual Heartbeat Rate', linestyle='--', color='r')
    ax[1].set_title('Heartbeat Rate')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Beats/min')
    ax[1].grid(True)

    ax[2].plot(results[2], label='Displacement')
    ax[2].set_title('Displacement')
    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('mm')
    ax[2].grid(True)
    
    logging.info(f"Range Resolution: {simulator.radar_config.range_resolution*1000:.2f} mm")
    logging.info(f"Maximum Range: {simulator.radar_config.max_range:.2f} m")
    logging.info(f"Velocity Resolution: {simulator.radar_config.velocity_resolution:.2f} m/s")


    plt.tight_layout()
    plt.show()
