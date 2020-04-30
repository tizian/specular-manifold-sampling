import os, sys
from subprocess import PIPE, run
import numpy as np

try:
    os.mkdir('results_kettle')
except:
    pass

def run_cmd(command, name):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results_kettle/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

def time_str(seconds):
    if seconds < 60:
        return "{:.2f} s".format(seconds)
    minutes = seconds // 60
    seconds %= 60
    if minutes < 60:
        return "{:.0f} min {:.2f} s".format(minutes, seconds)
    hours = minutes // 60
    minutes %= 60

    return "{:.0f} h {:.0f} min {:.2f} s".format(hours, minutes, seconds)

methods = [
    "pt",
    "sms_ub",
    "sms_b",
    "sms_bv",
    "yan"
]

method_names = [
    "path tracing",
    "unbiased SMS",
    "biased SMS",
    "biased + vectorized SMS",
    "the method of Yan et al. 2016"
]

if len(sys.argv) != 2:
    print("Need to provide a string argument to this script to specifiy the method to render.")
    sys.exit()

if not(sys.argv[1] in methods):
    print("Method string needs to be one of the following:")
    print(methods)
    sys.exit()

print("Render \"Kettle\" scene with {}:".format(method_names[methods.index(sys.argv[1])]))

N = 20
times = 10**np.linspace(0.0, 4.0, N)
for i in range(N):
    t = times[i]
    if sys.argv[1] == "yan":
        print("Render for {} .. (excluding pre-processing time)".format(time_str(t)))
    else:
        print("Render for {} ..".format(time_str(t)))

    if sys.argv[1] == "pt":
        name = "pt_time_{}".format(i)
        cmd = "mitsuba "
        cmd += "kettle_pt.xml "
        cmd += "-o results_kettle/{}.exr ".format(name)
        cmd += "-Dspp=999999999 "
        cmd += "-Dsamples_per_pass=1 "
        cmd += "-Dtimeout={} ".format(t)
        run_cmd(cmd, name)

    if sys.argv[1] == "sms_ub":
        name = "sms_ub_time_{}".format(i)
        cmd = "mitsuba "
        cmd += "kettle_sms.xml "
        cmd += "-o results_kettle/{}.exr ".format(name)
        cmd += "-Dspp=999999999 "
        cmd += "-Dsamples_per_pass=1 "
        cmd += "-Dtimeout={} ".format(t)
        cmd += "-Dglints_biased=false "
        cmd += "-Dglints_max_trials=100000 "
        run_cmd(cmd, name)

    if sys.argv[1] == "sms_b":
        name = "sms_b_time_{}".format(i)
        cmd = "mitsuba "
        cmd += "kettle_sms.xml "
        cmd += "-o results_kettle/{}.exr ".format(name)
        cmd += "-Dspp=999999999 "
        cmd += "-Dsamples_per_pass=1 "
        cmd += "-Dtimeout={} ".format(t)
        cmd += "-Dglints_biased=true "
        run_cmd(cmd, name)

    if sys.argv[1] == "sms_bv":
        name = "sms_bv_time_{}".format(i)
        cmd = "mitsuba "
        cmd += "kettle_sms.xml "
        cmd += "-o results_kettle/{}.exr ".format(name)
        cmd += "-Dspp=999999999 "
        cmd += "-Dsamples_per_pass=1 "
        cmd += "-Dtimeout={} ".format(t)
        cmd += "-Dglints_biased=true "
        cmd += "-Dglints_vectorized=true "
        run_cmd(cmd, name)

    if sys.argv[1] == "yan":
        name = "yan_time_{}".format(i)
        cmd = "mitsuba "
        cmd += "kettle_yan.xml "
        cmd += "-o results_kettle/{}.exr ".format(name)
        cmd += "-Dspp=999999999 "
        cmd += "-Dsamples_per_pass=1 "
        cmd += "-Dtimeout={} ".format(t)
        run_cmd(cmd, name)

print("done.")
