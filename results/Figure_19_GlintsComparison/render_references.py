import os, sys
from subprocess import PIPE, run
import numpy as np

try:
    os.mkdir('results_kettle')
except:
    pass

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results_shoes/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

# Shoes
spp = 100000

crop_s = 80

crop_1_x = 160
crop_1_y = 600

crop_2_x = 320
crop_2_y = 160

## CROP 1

cmd = "mitsuba "
cmd += "shoes_ref.xml "
cmd += "-o results_shoes/ref_crop_1.exr "
cmd += "-Dcrop_offset_x={} ".format(crop_1_x)
cmd += "-Dcrop_offset_y={} ".format(crop_1_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, "shoes_ref_crop_1")

## CROP 2

cmd = "mitsuba "
cmd += "shoes_ref.xml "
cmd += "-o results_shoes/ref_crop_2.exr "
cmd += "-Dcrop_offset_x={} ".format(crop_2_x)
cmd += "-Dcrop_offset_y={} ".format(crop_2_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, "shoes_ref_crop_2")


def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results_kettle/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

# Kettle
spp = 200000

crop_s = 80

crop_1_x = 330
crop_1_y = 400

crop_2_x = 390
crop_2_y = 200

## CROP 1

cmd = "mitsuba "
cmd += "kettle_ref.xml "
cmd += "-o results_kettle/ref_crop_1.exr "
cmd += "-Dcrop_offset_x={} ".format(crop_1_x)
cmd += "-Dcrop_offset_y={} ".format(crop_1_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, "kettle_ref_crop_1")

## CROP 2

cmd = "mitsuba "
cmd += "kettle_ref.xml "
cmd += "-o results_kettle/ref_crop_2.exr "
cmd += "-Dcrop_offset_x={} ".format(crop_2_x)
cmd += "-Dcrop_offset_y={} ".format(crop_2_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, "kettle_ref_crop_2")


print("done.")
