# Simulator for IWR6843 signals

Radar configurations are taken from an actual profile, see e.g. profiles/vital_signs_60ghz.cfg for an example .cfg file.

## Assumptions

- Single target, angle is approximately known, so angle estimation/beamforming 
- Stationary target, so Doppler processing is not needed
- Target distance is approximately known, and approximately 1 meter. Therefore CFAR (Constant False Alarm Rate) detection is not needed
- No clustering/tracking needed, so AoA (Angle of Arrival) processing can be skipped. However, for best phase measurements for vital signs, you may still want to use multiple RX antennas in an optimized way to improve SNR at the known subject position, rather than doing full spatial scanning with AoA processing.



## Digital signal processing components

### Range processing
- Use the Range DPU (Range Processing Unit) to perform 1D FFT on the ADC samples, to convert time domain samples to frequency domain (range bins). Since we know the approximate range, we only need to process the relevant range bin(s) around 1 meter.
- In the RangeDPU we select the range bins of interest:
    ```python
    # Calculate range bin around expected target location
    target_distance = self.signal_model.target.distance
    bin_start = max(0, int((target_distance - 0.2) / self.radar_config.range_resolution))
    bin_end = int((target_distance + 0.2) / self.radar_config.range_resolution)

    self.range_dpu = RangeDPU(
        num_adc_samples=self.radar_config.adc.num_samples,
        range_resolution=self.radar_config.range_resolution,
        range_bin_start=bin_start,  # We only look at bins near known distance
        range_bin_end=bin_end
    )
    ```


- Vital signs processing:
    - Extract phase information from the complex samples in the range bin of interest
    - Perform phase unwrapping
    - Apply bandpass filters to separate: Breathing signal (0.1-0.5 Hz typical) and Heartbeat signal (0.8-2.0 Hz typical)    

### No beamforming
- The xWR68xx devices use a fixed antenna array pattern and don't have active/electronic beam steering capability
- Based on the SDK documentation and device capabilities, digital beamforming with the IWR6843/xWR68xx could be implemented using the MIMO virtual array. One would enable multiple TX and RX antennas through channelCfg and configure TDM-MIMO through chirpCfg where each chirp uses a different TX antenna. For each range bin, you would have 12 virtual antennas (3 TX × 4 RX), then complex samples from these virtual antennas can be combined using steering vectors. For now this option is not investigated.

