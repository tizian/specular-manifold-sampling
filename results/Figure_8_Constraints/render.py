import os
from subprocess import PIPE, run
from PIL import Image, ImageDraw, ImageFont

def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

def run_cmd(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

def process_output(out):
    out = find_between(out, "Specular Manifold Sampling Statistics", "[Profiler]")
    # print(out)

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

    return (int(num_succeeded), float(per_succeeded)), (int(num_failed), float(per_failed)), float(avg_booth)

try:
    os.mkdir('results')
except:
    pass

time = 10.0

## SPHERE

print("Render half-vector version ..")
cmd = "mitsuba sphere.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_halfvector_constraints=true -o results/sphere_halfvector.exr".format(time)
out_sphere_hv = process_output(run_cmd(cmd))

print("Render angle-difference version ..")
cmd = "mitsuba sphere.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_halfvector_constraints=false -o results/sphere_anglediff.exr".format(time)
out_sphere_dir = process_output(run_cmd(cmd))

## SPHERE

with open('results/log.txt', 'w') as file:
    file.write("Equal time comparison, {} s\n\n".format(time))

    file.write("Sphere\n\n")

    file.write("halfvector\n")
    file.write("Succeeded:\t{}\t{}%\n".format(out_sphere_hv[0][0], out_sphere_hv[0][1]))
    file.write("Failed   :\t{}\t{}%\n".format(out_sphere_hv[1][0], out_sphere_hv[1][1]))
    file.write("avg Booth its:\t{}\n".format(out_sphere_hv[2]))
    file.write("\n")

    file.write("anglediff\n")
    file.write("Succeeded:\t{}\t{}%\n".format(out_sphere_dir[0][0], out_sphere_dir[0][1]))
    file.write("Failed   :\t{}\t{}%\n".format(out_sphere_dir[1][0], out_sphere_dir[1][1]))
    file.write("avg Booth its:\t{}\n".format(out_sphere_dir[2]))
    file.write("\n\n\n")

print("done.")