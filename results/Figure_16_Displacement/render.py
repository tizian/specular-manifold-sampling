import os
from subprocess import PIPE, run

def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

def run_cmd(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

def process_output(out):

    spp = -1
    lines = out.splitlines(True)
    for line in lines:
        if "Rendering finished. Computed" in line:
            tokens = line.split()
            spp = tokens[8]

    out = find_between(out, "Specular Manifold Sampling Statistics", "[Profiler]")

    lines = out.splitlines(True)
    for line in lines:
        if "Solver succeeded" in line:
            tokens = line.split()
            num_succeeded = tokens[2]
            per_succeeded = tokens[3][1:-2]
        elif "Solver failed" in line:
            tokens = line.split()
            num_failed = tokens[2]
            per_failed = tokens[3][1:-2]
        elif "avg. Booth" in line:
            tokens = line.split()
            avg_booth = tokens[3]

    return (int(num_succeeded), float(per_succeeded)), (int(num_failed), float(per_failed)), float(avg_booth), int(spp)


try:
    os.mkdir('results')
except:
    pass

time = 2

names = []
statistics = []

for i in range(3):
    print("Render displaced version {} ..".format(i))
    name = "sphere_{}_disp".format(i+1)
    cmd = "mitsuba {}.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -o results/{}.exr".format(name, time, name)
    stats = process_output(run_cmd(cmd))

    names.append(name)
    statistics.append(stats)

for i in range(3):
    print("Render normal-mapped version {} ..".format(i))
    name = "sphere_{}_nrm".format(i+1)
    cmd = "mitsuba {}.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -o results/{}.exr".format(name, time, name)
    stats = process_output(run_cmd(cmd))

    names.append(name)
    statistics.append(stats)


with open('results/log.txt', 'w') as file:
    file.write("Equal time comparison, {} s\n\n".format(time))

    for idx in range(len(names)):
        name = names[idx]
        stats = statistics[idx]

        file.write("{}\n\n".format(name))
        file.write("Succeeded:\t{}\t{}%\n".format(stats[0][0], stats[0][1]))
        file.write("Failed   :\t{}\t{}%\n".format(stats[1][0], stats[1][1]))
        file.write("avg Booth its:\t{}\n".format(stats[2]))
        file.write("spp:\t{}\n".format(stats[3]))
        file.write("\n")
