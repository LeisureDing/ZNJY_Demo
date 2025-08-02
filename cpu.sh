#!/bin/bash

echo "userspace" > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed

echo "userspace" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2016000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed

echo "userspace" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo 2016000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed

echo "CPU0-3 current frequency: $(cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq) Hz"
echo "CPU4-5 current frequency: $(cat /sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq) Hz"
echo "CPU6-7 current frequency: $(cat /sys/devices/system/cpu/cpufreq/policy6/cpuinfo_cur_freq) Hz"

echo "Running Python script"
python3 /home/orangepi/SmartSave/Saving_demo/Controller.py