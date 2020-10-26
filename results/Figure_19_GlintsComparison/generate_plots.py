try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import Bitmap, Struct

import os, datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.markers

scene_names = ["shoes",
               "kettle"]
scene_crops = [(160, 600, 320, 160),
               (330, 400, 390, 200)]
sizeof_flake = 160 # bytes

def process_output(lines):
    data = {}
    for line in lines:
        if "[SamplingIntegrator] Rendering finished." in line:
            tokens = line.split()
            data['spp']  = tokens[8]
            data['time'] = tokens[12][:-1]
        elif "[YanGlintyBSDF] done." in line:
            tokens = line.split()
            data['time_preprocess'] = tokens[7][:-1]
        elif "Hierarchy size" in line and not 'hierarchy_size' in data:
            tokens = line.split()
            data['hierarchy_size'] = tokens[5]
    return data

def error(ref, img):
    tmp = np.sqrt(np.mean((img-ref)**2))
    if tmp == np.inf:
        # Huge outliers in unbiased method cause issues for plotting
        return 100
    return tmp

def time_errors(method, scene_idx):
    scene_name = scene_names[scene_idx]
    crop_s = 80
    crop_1_x, crop_1_y, crop_2_x, crop_2_y = scene_crops[scene_idx]
    path = "results_{}".format(scene_name)

    N = 20
    times = 10**np.linspace(0, 4, N)

    images = []
    for i in range(len(times)):
        img = np.array(Bitmap("{}/{}_time_{}.exr".format(path, method, i)))

        crops = (img[crop_1_y:crop_1_y+crop_s, crop_1_x:crop_1_x+crop_s],
                 img[crop_2_y:crop_2_y+crop_s, crop_2_x:crop_2_x+crop_s],
                 img)
        images.append(crops)

    ref_1, ref_2, ref_full = images[-1]
    error_1 = []; error_2 = []; error_full = []
    for i in range(N-1):
        cur_1, cur_2, cur_full = images[i]
        error_1.append(error(ref_1, cur_1))
        error_2.append(error(ref_2, cur_2))
        error_full.append(error(ref_full, cur_full))

    return error_1, error_2, error_full

def process(scene_idx):
    scene_name = scene_names[scene_idx]
    crop_s = 80
    crop_1_x, crop_1_y, crop_2_x, crop_2_y = scene_crops[scene_idx]
    path = "results_{}".format(scene_name)
    print("Processing scene: \"{}\"".format(scene_name))

    N = 20
    times = 10**np.linspace(0, 4, N)

    imgs_sms = []
    imgs_yan = []

    print("idx          time      spp (yan)      spp (sms_ub)      spp (sms_b)      spp (sms_bv)")
    print("=========================================================================================")
    for i in range(len(times)):
        with open("{}/log_yan_time_{}.txt".format(path, i)) as f:
            log_yan = f.readlines()
        data_yan = process_output(log_yan)

        with open("{}/log_sms_ub_time_{}.txt".format(path, i)) as f:
            log_sms_ub = f.readlines()
        data_sms_ub = process_output(log_sms_ub)

        with open("{}/log_sms_b_time_{}.txt".format(path, i)) as f:
            log_sms_b = f.readlines()
        data_sms_b = process_output(log_sms_b)

        with open("{}/log_sms_bv_time_{}.txt".format(path, i)) as f:
            log_sms_bv = f.readlines()
        data_sms_bv = process_output(log_sms_bv)

        time       = data_yan['time']
        spp_yan    = data_yan['spp']
        spp_sms_ub = data_sms_ub['spp']
        spp_sms_b  = data_sms_b['spp']
        spp_sms_bv = data_sms_bv['spp']

        print("{:>3}      {:>8}         {:>6}            {:>6}           {:>6}           {:>6}".format(i, time, spp_yan, spp_sms_ub, spp_sms_b, spp_sms_bv))

    print("")
    print("Yan preprocess time: {}".format(data_yan['time_preprocess']))
    hierarchy_size = data_yan['hierarchy_size']
    hierarchy_size_gb = int(hierarchy_size) * sizeof_flake / 1024 / 1024 / 1024
    print("Yan hierarchy size: {} = {} GB".format(hierarchy_size, hierarchy_size_gb))



    show_vectorized = True

    methods = ['yan', 'sms_ub', 'sms_b', 'sms_bv']
    names   = ['Yan [2016]', 'Ours (unbiased)', 'Ours (biased)', 'Ours (biased+vec)']
    lines   = ['k', 'b', 'r', 'g']
    lws     = [2, 1, 1, 1]

    fig, ax = plt.subplots(ncols=2, figsize=(20,3))
    for i in range(2):
        ax[i].loglog(); ax[i].grid(True, which="major");
        if scene_idx == 0:
            ax[i].set_ylim([0.002, 1])
        elif scene_idx == 1:
            ax[i].set_ylim([0.008, 5])
        elif scene_idx == 2:
            ax[i].set_ylim([0.002, 1])
        ax[i].tick_params(axis='both', which='major', labelsize=15)
        labels = ax[i].get_yticklabels()
        for lb in labels:
            lb.set_verticalalignment('top')

    idx = 0
    for method, name, line, lw in zip(methods, names, lines, lws):
        errors = time_errors(method, scene_idx)

        ax[0].plot(times[:-1], errors[0],
                   '{}'.format(line), ms=4, label="{}".format(name), marker='o', lw=lw)
        ax[1].plot(times[:-1], errors[1],
                   '{}'.format(line), ms=4, label="{}".format(name), marker='o', lw=lw)

        if idx == 2 and not show_vectorized:
            break
        idx += 1

    ax[0].legend(ncol=1, loc='upper right')
    ax[1].legend(ncol=1, loc='upper right')

    # plt.show()

    extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    name = "{}/plot_{}_1.pdf".format(path, scene_names[scene_idx])
    fig.savefig(name, bbox_inches=extent.expanded(1.2, 1.3), transparent=True)

    extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    name = "{}/plot_{}_2.pdf".format(path, scene_names[scene_idx])
    fig.savefig(name, bbox_inches=extent.expanded(1.2, 1.3), transparent=True)


process(0)
process(1)
