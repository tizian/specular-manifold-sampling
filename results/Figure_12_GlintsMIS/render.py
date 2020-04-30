import os
from subprocess import PIPE, run

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

try:
    os.mkdir('results')
except:
    pass

spp = 512

## BSDF-ONLY

name = "curved_bsdf"
cmd = "mitsuba "
cmd += "curved.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dglints_bsdf_strategy_only=true "
cmd += "-Dglints_sms_strategy_only=false "
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, name)


## SMS-ONLY

name = "curved_sms"
cmd = "mitsuba "
cmd += "curved.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dglints_bsdf_strategy_only=false "
cmd += "-Dglints_sms_strategy_only=true "
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, name)


## COMBINED

name = "curved_combined"
cmd = "mitsuba "
cmd += "curved.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dglints_bsdf_strategy_only=false "
cmd += "-Dglints_sms_strategy_only=false "
cmd += "-Dspp={} ".format(spp)
run_cmd(cmd, name)
