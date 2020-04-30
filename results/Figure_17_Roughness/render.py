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

time = 1.0

for alpha in [0.005, 0.02, 0.1]:
    name = "plane_sms_{}".format(alpha)
    cmd = "mitsuba plane_sms.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_twostage=true -Dalpha={} -o results/{}.exr".format(time, alpha, name)
    run_cmd(cmd, name)

    name = "plane_pt_{}".format(alpha)
    cmd = "mitsuba plane_pt.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dalpha={} -o results/{}.exr".format(time, alpha, name)
    run_cmd(cmd, name)
