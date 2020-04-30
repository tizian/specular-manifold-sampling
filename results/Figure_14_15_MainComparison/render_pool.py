import os
from subprocess import PIPE, run

try:
    os.mkdir('results_pool')
except:
    pass

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results_pool/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

# 5 min for all methods that we want to compare
timeout = 5*60

name = "pool_pt"
cmd = "mitsuba "
cmd += "pool_pt.xml "
cmd += "-o results_pool/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
run_cmd(cmd, name)

name = "pool_mnee"
cmd = "mitsuba "
cmd += "pool_sms.xml "
cmd += "-o results_pool/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dcaustics_mnee_init=true "
cmd += "-Dcaustics_max_trials=1 "
run_cmd(cmd, name)

name = "pool_mnee_biased"
cmd = "mitsuba "
cmd += "pool_sms.xml "
cmd += "-o results_pool/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dcaustics_mnee_init=true "
cmd += "-Dcaustics_max_trials=1 "
cmd += "-Dbiased_mnee=true "
run_cmd(cmd, name)

name = "pool_sms_ub"
cmd = "mitsuba "
cmd += "pool_sms.xml "
cmd += "-o results_pool/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_twostage=true "
run_cmd(cmd, name)

for M in [1, 2, 4, 8]:
    name = "pool_sms_b{:02d}".format(M)
    cmd = "mitsuba "
    cmd += "pool_sms.xml "
    cmd += "-o results_pool/{}.exr ".format(name)
    cmd += "-Dspp=999999999 "
    cmd += "-Dsamples_per_pass=1 "
    cmd += "-Dtimeout={} ".format(timeout)
    cmd += "-Dcaustics_twostage=true "
    cmd += "-Dcaustics_biased=true "
    cmd += "-Dcaustics_max_trials={} ".format(M)
    run_cmd(cmd, name)

print("done.")
