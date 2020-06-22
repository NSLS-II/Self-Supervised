from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os
import re
import numpy as np
from optparse import OptionParser
from multiprocessing import Pool
import mrcfile


from skimage.util import view_as_windows
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from skimage.transform import resize

from dataLoader import DataLoader
from starReader import starRead

## write coordinates to a star file
def write_coordinate(coordinate, mrc_file, coordinate_suffix, output_dir):
    mrc_basename = os.path.basename(mrc_file)
    print(mrc_basename)
    coordinate_name = os.path.join(
        output_dir, mrc_basename[:-4] + coordinate_suffix + ".star")
    print(coordinate_name)
    f = open(coordinate_name, 'w')
    f.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
    for i in range(len(coordinate)):
        f.write(str(coordinate[i][0]) + ' ' + str(coordinate[i][1]) + '\n')
    f.close()


def image2Peaks(image2d, distance, threshold):
    # find local maxima and output a binary array
    blobs = peak_local_max(image2d,
                           min_distance=1,
                           indices=False,
                           exclude_border=False)
    # label the array for all non zero values, output array with labels.
    labeled, num_objects = ndimage.label(blobs)
    # Center of mass  to find coordinate
    center2d = np.array(
        ndimage.center_of_mass(image2d, labeled, range(1, num_objects + 1)))
    center2d = center2d.astype(int)
    list_y_x = center2d.tolist()
    for i in range(len(list_y_x)):
        # append classes values 1 or 0 to the list, column 3
        list_y_x[i].append(image2d[center2d[i][0]][center2d[i][1]])
        # append 0 to the list, column 4
        list_y_x[i].append(0)
    # distance cutoff between particles, 0 keep, 1 remove.
    for i in range(len(list_y_x) - 1):
        if list_y_x[i][3] == 1:
            continue

        for j in range(i + 1, len(list_y_x)):
            if list_y_x[i][3] == 1:
                break
            if list_y_x[j][3] == 1:
                continue
            d_y = list_y_x[i][0] - list_y_x[j][0]
            d_x = list_y_x[i][1] - list_y_x[j][1]
            d_distance = math.sqrt(d_y**2 + d_x**2)
            if d_distance < distance:
                if list_y_x[i][2] >= list_y_x[j][2]:
                    list_y_x[j][3] = 1
                else:
                    list_y_x[i][3] = 1
    list_coordinates = []
    for i in range(len(list_y_x)):
        if list_y_x[i][3] == 0 and list_y_x[i][2] > threshold:
            # remove the symbol element
            list_x_y = []
            list_x_y.append(list_y_x[i][1])
            list_x_y.append(list_y_x[i][0])
            list_x_y.append(list_y_x[i][2])
            list_coordinates.append(list_x_y)

    return list_coordinates


def pick_mp(args):
    i, mrc_file, model_input_size, particle_size, coordinate_suffix, threshold, output_dir, bin_size = args
    time1 = time.time()
    ## for Keras/tensorflow configuration
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
#    gpulist = ['0', '1', '2', '3']
#    if str(i % 4) in gpulist:
#        os.environ["CUDA_VISIBLE_DEVICES"] = str(i % 4)
#        print("use GPU for micrograph # ", i, i % 4)
#    else:
#        os.environ["CUDA_VISIBLE_DEVICES"] = ""
#        print("use CPU for micrograph # ", i)
    # initialize the model
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(i % 4)
    
    ### CPU only. 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from keras.optimizers import SGD
    from keras.models import load_model

    model=load_model('test_model.h5')
    model.compile(optimizer=SGD(0.01),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # read the micrograph image data
    print(mrc_file)
    with mrcfile.open(mrc_file, mode='r', permissive=True) as mrc:
        # mrc.header.map = mrcfile.constants.MAP_ID
        header = mrc.header
        body_2d = mrc.data
    # preprocess  micrograph
    print("raw shape: ", body_2d.shape[0], body_2d.shape[1])
    body_2d = DataLoader.preprocess_micrograph_picker(body_2d, bin_size)
    step_size = 4
    patch_size = int(round(particle_size/bin_size))
    d_min = int(round(0.5*patch_size/4.))
    # patches extraction using rolling windows
    # https://gist.github.com/hasnainv/49dc4a85933de6b979f8
    # window_shape = (patch_size, patch_size)
    window_shape = (64, 64)
    patches = view_as_windows(body_2d, window_shape, step_size)
    nR, nC, H, W = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W))
    #print("shape of extracted pathces: ", patches.shape, nR, nC)
    patches = (patches - patches.mean(axis=(1, 2), keepdims=1)) / patches.std(
        axis=(1, 2), keepdims=1)
    patches = patches.reshape(nWindow, model_input_size[1],
                              model_input_size[2], 1)
    proba = model.predict_proba(patches, batch_size=500)
    classes = proba[:, 1].reshape(nR, nC)
    time_cost = time.time() - time1
    list_coordinate = image2Peaks(classes, d_min, threshold)
    time_cost = time.time() - time1
    print("Particles:", len(list_coordinate),",", "time cost final: %d s"%time_cost)
    for i in range(len(list_coordinate)):
        list_coordinate[i].append(mrc_file)
        # scale the coordinates to raw image
        list_coordinate[i][0] = (list_coordinate[i][0]*step_size+patch_size/2) * bin_size
        list_coordinate[i][1] = (list_coordinate[i][1]*step_size+patch_size/2) * bin_size
    write_coordinate(list_coordinate, mrc_file, coordinate_suffix, output_dir)
    return list_coordinate

def main():
    # define the options
    parser = OptionParser()
    parser.add_option("--input_dir",
                      dest="input_dir",
                      help="Input directory",
                      metavar="DIRECTORY")
    parser.add_option("--pre_trained_model",
                      dest="pre_trained_model",
                      help="Input the pre-trained model",
                      default="test_model.h5",
                      metavar="FILE")
    parser.add_option("--star_file",
                      dest="star_file",
                      help="Micrograph star file for picking",
                      metavar="FILE")
    parser.add_option("--particle_size",
                      dest="particle_size",
                      type="int",
                      help="the size of the particle in pixels.",
                      metavar="VALUE",
                      default=150)
    parser.add_option("--bin_size",
                      dest="bin_size",
                      type="int",
                      help="image size reduction",
                      metavar="VALUE",
                      default=4)

    parser.add_option("--threads",
                      dest="threads",
                      type="int",
                      help="how many processors to use",
                      metavar="VALUE",
                      default=1)
    parser.add_option("--output_dir",
                      dest="output_dir",
                      help="Output directory, the coordinates file will be saved here.",
                      metavar="DIRECTORY")
    parser.add_option("--coordinate_suffix",
                      dest="coordinate_suffix",
                      help="The suffix of picked coordinate file, like '_kpick'",
                      metavar="STRING")
    parser.add_option("--threshold",
                      dest="threshold",
                      type="float",
                      help="Prediction probability of being a particle",
                      metavar="VALUE",
                      default=0.7)
    parser.add_option("--testRun",
                      dest="testRun",
                      default=False,
                      action="store_true",
                      help="Prediction probability of being a particle")
    (opt, args) = parser.parse_args()

    # define the input size to the model
    model_input_size = [1000, 64, 64, 1]

    if not os.path.isfile(opt.pre_trained_model):
        print("ERROR:%s is not a valid file." % (opt.pre_trained_model))

    if not os.path.isdir(opt.input_dir):
        print("ERROR:%s is not a valid dir." % (opt.input_dir))

    if not os.path.isdir(opt.output_dir):
        os.mkdir(opt.output_dir)

    # load mrc files, ignore micrographs already picked.
    mrc_file_all = []
    if opt.star_file:
        micrograh_star = starRead(opt.star_file)
        table_star = micrograh_star.getByName('data_')
        mrc_list = table_star.getByName('_rlnMicrographName')
        starfiles = os.listdir(opt.output_dir)
        for f in mrc_list:
            fname = f[8:]
            flag = True
            for j in starfiles:
                if fname == j[:-13]:
                    flag = False
                    break
            if flag:
                mrc_file_all.append(f)

    else:
        files = os.listdir(opt.input_dir)
        starfiles = os.listdir(opt.output_dir)
        for f in files:
            if re.search('.mrc', f):
                flag = True
                fname = f[:-4]
                for j in starfiles:
                    if fname == j[:-13]:
                        flag = False
                        break
                if flag:
                    filename = os.path.join(opt.input_dir, f)
                    mrc_file_all.append(filename)

    mrc_file_all.sort()
    mrc_number = len(mrc_file_all)
    if opt.testRun is True:
        mrc_number = 10
    time1 = time.time()
    print("number of micrographs used: ", mrc_number, opt.testRun)
    pl = Pool(opt.threads)
    jobs = []
    for i in range(mrc_number):
        jobs.append([
            i, mrc_file_all[i], model_input_size, opt.particle_size,
            opt.coordinate_suffix, opt.threshold, opt.output_dir, opt.bin_size  ])
    results = pl.map(pick_mp, jobs)
    candidate_particle_all = []
    for r in results:
        candidate_particle_all.append(r)
    pl.close()
    pl.join()
    time_cost = time.time() - time1
    print("total time cost: %.1f s" % (time_cost))


if __name__ == '__main__':
    main()
