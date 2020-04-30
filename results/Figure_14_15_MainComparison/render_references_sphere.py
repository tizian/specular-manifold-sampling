import os
from subprocess import PIPE, run

try:
    os.mkdir('results_sphere')
except:
    pass

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results_sphere/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

angle = 60.0

# 8 hours for ours reference
timeout = 8*60*60

name = "sphere_ref_sms"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-Dlight_angle={} ".format(angle)
cmd += "-o results_sphere/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_twostage=true "
run_cmd(cmd, name)

# 8 hours for pt reference
timeout = 8*60*60

name = "sphere_ref_pt"
cmd = "mitsuba "
cmd += "sphere_pt.xml "
cmd += "-Dlight_angle={} ".format(angle)
cmd += "-o results_sphere/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
run_cmd(cmd, name)

# 8 hours for path traced reference insets
timeout = 8*60*60

crop_s = 200
crop_1_x = 920
crop_1_y = 400

name = "sphere_ref_pt_inset_1"
cmd = "mitsuba "
cmd += "sphere_pt.xml "
cmd += "-Dlight_angle={} ".format(angle)
cmd += "-o results_sphere/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcrop_offset_x={} ".format(crop_1_x)
cmd += "-Dcrop_offset_y={} ".format(crop_1_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
run_cmd(cmd, name)

crop_2_x = 620
crop_2_y = 160

name = "sphere_ref_pt_inset_2"
cmd = "mitsuba "
cmd += "sphere_pt.xml "
cmd += "-Dlight_angle={} ".format(angle)
cmd += "-o results_sphere/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcrop_offset_x={} ".format(crop_2_x)
cmd += "-Dcrop_offset_y={} ".format(crop_2_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
run_cmd(cmd, name)

print("done.")
