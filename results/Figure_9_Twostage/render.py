import os
from subprocess import PIPE, run
from PIL import Image, ImageDraw, ImageFont

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

time = 60.0 # seconds

## PLANE

print("Render plane (one-stage) ..")
cmd = "mitsuba plane.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_twostage=false -o results/plane_1.exr".format(time)
out_plane_1 = process_output(run_cmd(cmd))

print("Render plane (two-stage) ..")
cmd = "mitsuba plane.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_twostage=true  -o results/plane_2.exr".format(time)
out_plane_2 = process_output(run_cmd(cmd))

## SPHERE

print("Render sphere (one-stage) ..")
cmd = "mitsuba sphere.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_twostage=false -o results/sphere_1.exr".format(time)
out_sphere_1 = process_output(run_cmd(cmd))

print("Render sphere (two-stage) ..")
cmd = "mitsuba sphere.xml -Dsamples_per_pass=1 -Dspp=9999 -Dtimeout={} -Dcaustics_twostage=true  -o results/sphere_2.exr".format(time)
out_sphere_2 = process_output(run_cmd(cmd))


with open('results/log.txt', 'w') as file:
    file.write("Equal time comparison, {} s\n\n".format(time))

    file.write("Plane\n\n")

    file.write("one-stage\n")
    file.write("spp: {}\n".format(out_plane_1[3]))
    file.write("Succeeded:\t{}\t{}%\n".format(out_plane_1[0][0], out_plane_1[0][1]))
    file.write("Failed   :\t{}\t{}%\n".format(out_plane_1[1][0], out_plane_1[1][1]))
    file.write("avg Booth its:\t{}\n".format(out_plane_1[2]))
    file.write("\n")

    file.write("two-stage\n")
    file.write("spp: {}\n".format(out_plane_2[3]))
    file.write("Succeeded:\t{}\t{}%\n".format(out_plane_2[0][0], out_plane_2[0][1]))
    file.write("Failed   :\t{}\t{}%\n".format(out_plane_2[1][0], out_plane_2[1][1]))
    file.write("avg Booth its:\t{}\n".format(out_plane_2[2]))
    file.write("\n\n\n")

    file.write("Sphere\n\n")

    file.write("one-stage\n")
    file.write("spp: {}\n".format(out_sphere_1[3]))
    file.write("Succeeded:\t{}\t{}%\n".format(out_sphere_1[0][0], out_sphere_1[0][1]))
    file.write("Failed   :\t{}\t{}%\n".format(out_sphere_1[1][0], out_sphere_1[1][1]))
    file.write("avg Booth its:\t{}\n".format(out_sphere_1[2]))
    file.write("\n")

    file.write("two-stage\n")
    file.write("spp: {}\n".format(out_sphere_2[3]))
    file.write("Succeeded:\t{}\t{}%\n".format(out_sphere_2[0][0], out_sphere_2[0][1]))
    file.write("Failed   :\t{}\t{}%\n".format(out_sphere_2[1][0], out_sphere_2[1][1]))
    file.write("avg Booth its:\t{}\n".format(out_sphere_2[2]))
    file.write("\n\n\n")

print("done.")