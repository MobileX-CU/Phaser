# Phaser 
This repo contains all code for the paper Set Phasers to Stun: Beaming Power and Control to Mobile Robots with Laser Light (link coming soon!). We also provide a detailed guide for replicating the Phaser tracking and laser steering design, which enables integration of 3D object tracking via stereo-vision with any laser steering device. 

## Contents
Resources in this repo are organized into three folders:

1. [track_steer](track_steer) Code, Jupyter notebooks, and example data for calibrating and deploying a stereo-vision-based tracking and laser steering system using Phaser methodologies. We also include a detailed guide on adapting our provided code and instructions to any tracking and laser steering system using stereo-vision, to support future works and laser-based applications.
2. [laser_modulation](laser_modulation): Code used for FSK current-modulation of laser light in our evaluation.
3. [millimobile](millimobile): Firmware used in the paper evaluation to receive laser data and respond to laser-provided commands onboard MilliMobile robots.
