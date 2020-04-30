import os
from subprocess import PIPE, run

try:
    os.mkdir('results')
except:
    pass

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

# 5 min for all methods that we want to compare
timeout = 5*60

crop_s = 1080
crop_x = 420
crop_y = 0

name = "slab_pt"
cmd = "mitsuba "
cmd += "slab_pt.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcrop_offset_x={} ".format(crop_x)
cmd += "-Dcrop_offset_y={} ".format(crop_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
run_cmd(cmd, name)

name = "slab_mnee"
cmd = "mitsuba "
cmd += "slab_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcrop_offset_x={} ".format(crop_x)
cmd += "-Dcrop_offset_y={} ".format(crop_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dcaustics_mnee_init=true "
cmd += "-Dcaustics_max_trials=1 "
run_cmd(cmd, name)

name = "slab_mnee_biased"
cmd = "mitsuba "
cmd += "slab_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcrop_offset_x={} ".format(crop_x)
cmd += "-Dcrop_offset_y={} ".format(crop_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dcaustics_mnee_init=true "
cmd += "-Dcaustics_max_trials=1 "
cmd += "-Dbiased_mnee=true "
run_cmd(cmd, name)

name = "slab_sms_ub"
cmd = "mitsuba "
cmd += "slab_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcrop_offset_x={} ".format(crop_x)
cmd += "-Dcrop_offset_y={} ".format(crop_y)
cmd += "-Dcrop_width={} ".format(crop_s)
cmd += "-Dcrop_height={} ".format(crop_s)
run_cmd(cmd, name)

for M in [1, 2, 4, 8]:
    name = "slab_sms_b{:02d}".format(M)
    cmd = "mitsuba "
    cmd += "slab_sms.xml "
    cmd += "-o results/{}.exr ".format(name)
    cmd += "-Dspp=999999999 "
    cmd += "-Dsamples_per_pass=1 "
    cmd += "-Dtimeout={} ".format(timeout)
    cmd += "-Dcrop_offset_x={} ".format(crop_x)
    cmd += "-Dcrop_offset_y={} ".format(crop_y)
    cmd += "-Dcrop_width={} ".format(crop_s)
    cmd += "-Dcrop_height={} ".format(crop_s)
    cmd += "-Dcaustics_biased=true "
    cmd += "-Dcaustics_max_trials={} ".format(M)
    run_cmd(cmd, name)

print("done.")
