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

# 5 hours for path traced reference
timeout = 5*60*60

name = "sphere_ref_pt"
cmd = "mitsuba "
cmd += "sphere_pt.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
run_cmd(cmd, name)


# 5 hour for SMS reference
timeout = 5*60*60

name = "sphere_ref_sms"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_twostage=true "
run_cmd(cmd, name)

print("done.")
