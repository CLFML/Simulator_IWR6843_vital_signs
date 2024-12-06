radar_simulator/
│
├── config/
│   ├── __init__.py
│   ├── radar_config.py         # RadarConfig dataclass
│   ├── human_target.py         # HumanTarget dataclass
│   ├── config_parser.py        # .cfg file parser
│   └── config_validator.py     # Configuration validation
│
├── core/
│   ├── __init__.py
│   ├── signal_model.py         # Signal generation 
│   └── vital_signs_model.py    # Vital signs modeling
│
├── utils/
│   ├── __init__.py
│   └── logger.py              # Logging configuration
│
├── profiles/                   # Example .cfg files
│   └── vital_signs_60ghz.cfg
│
├── logs/                      # Log output directory
│
├── main.py
└── requirements.txt