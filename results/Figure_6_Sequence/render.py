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

# # 1 min for all methods that we want to compare
timeout = 1*60

# MNEE, normal
name = "sphere_mnee"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_max_trials=1 "
cmd += "-Dcaustics_mnee_init=true "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dbiased_mnee=false "
run_cmd(cmd, name)

# MNEE, biased to remove noise
name = "sphere_mnee_biased"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_max_trials=1 "
cmd += "-Dcaustics_mnee_init=true "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dbiased_mnee=true "
run_cmd(cmd, name)


# Unbiased versions
name = "sphere_sms_ub"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "
cmd += "-Dcaustics_max_trials=1000000000 "
cmd += "-Dcaustics_mnee_init=false "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dcaustics_twostage=false "
run_cmd(cmd, name)

name = "sphere_sms_ub_constr"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "
cmd += "-Dcaustics_max_trials=1000000000 "
cmd += "-Dcaustics_mnee_init=false "
cmd += "-Dcaustics_halfvector_constraints=false "
cmd += "-Dcaustics_twostage=false "
run_cmd(cmd, name)

name = "sphere_sms_ub_constr_twostage"
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "
cmd += "-Dcaustics_max_trials=1000000000 "
cmd += "-Dcaustics_mnee_init=false "
cmd += "-Dcaustics_halfvector_constraints=false "
cmd += "-Dcaustics_twostage=true "
run_cmd(cmd, name)


# Biased versions, with Bernoulli trial set size 16
M = 16

name = "sphere_sms_b{}".format(M)
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_max_trials={} ".format(M)
cmd += "-Dcaustics_mnee_init=false "
cmd += "-Dcaustics_halfvector_constraints=true "
cmd += "-Dcaustics_twostage=false "
run_cmd(cmd, name)

name = "sphere_sms_b{}_constr".format(M)
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_max_trials={} ".format(M)
cmd += "-Dcaustics_mnee_init=false "
cmd += "-Dcaustics_halfvector_constraints=false "
cmd += "-Dcaustics_twostage=false "
run_cmd(cmd, name)

name = "sphere_sms_b{}_constr_twostage".format(M)
cmd = "mitsuba "
cmd += "sphere_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=true "
cmd += "-Dcaustics_max_trials={} ".format(M)
cmd += "-Dcaustics_mnee_init=false "
cmd += "-Dcaustics_halfvector_constraints=false "
cmd += "-Dcaustics_twostage=true "
run_cmd(cmd, name)

print("done.")
