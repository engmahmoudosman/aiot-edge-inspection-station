#!/bin/bash
cd ~/hailo-apps
source setup_env.sh
cd ~/inspection_station
python src/sensor_fusion_m3.py
