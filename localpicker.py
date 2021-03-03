from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
from optparse import OptionParser
import mrcfile
from dataLoader import DataLoader
import matplotlib.pyplot as plt
from skimage.filters import threshold_local, median
from skimage.feature import blob_doh, peak_local_max
from skimage.feature import canny
#from least_square_circle import leastsq_circle
from scipy import ndimage 

def local_max(img, labels, box_size=20):
    list_pks = labels.tolist()
    for i in range(len(list_pks)):
        list_pks[i].append(1)
    # distance cutoff between particles, 1 keep, 0 remove.
    for i in range(len(list_pks) - 1):
        if list_pks[i][3] == 0:
            continue

        for j in range(i + 1, len(list_pks)):
            if list_pks[j][3] == 0:
                continue
            y = list_pks[i][0] - list_pks[j][0]
            x = list_pks[i][1] - list_pks[j][1]
            distance = math.sqrt(x**2 + y**2)
            if distance < box_size / 2.0 and list_pks[i][2] > list_pks[j][2]:
                list_pks[j][3] = 0
            if distance < box_size / 2.0 and list_pks[i][2] <= list_pks[j][2]:
                list_pks[i][3] = 0
    list_coordinates = []
    for i in range(len(list_pks)):
        if list_pks[i][3] == 1:
            if list_pks[i][1] - int(box_size / 2) > 0 and list_pks[i][1] + int(
                    box_size / 2) < img.shape[1] and list_pks[i][0] - int(
                        box_size / 2) > 0 and list_pks[i][0] + int(
                            box_size / 2) < img.shape[0]:              
                list_selected = []
                list_selected.append(list_pks[i][1])
                list_selected.append(list_pks[i][0])
                list_coordinates.append(list_selected)
    return list_coordinates


def edge_remove(image):
    pass

def ice_detect(image):
    pass

def radial_profile(image,center):
    pass

def write_coordinate(coordinate, mrc_file, coordinate_suffix, dirname):
    mrc_basename = os.path.basename(mrc_file)
    coordinate_name = os.path.join(
        dirname, mrc_basename[:-4] + coordinate_suffix + ".star")
    print(coordinate_name, " # of particles: ", len(coordinate))

    if len(coordinate) > 5:  # quality control check
        f = open(coordinate_name, 'w')
        f.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
        for i in range(len(coordinate)):
            f.write(str(coordinate[i][0]) + ' ' + str(coordinate[i][1]) + '\n')
        f.close()


def localPicker():
    parser = OptionParser()
    parser.add_option("--mrc_file",
                      dest="mrc_file",
                      help="mrc file name",
                      metavar="FILE")
    parser.add_option("--step_size",
                      dest="step_size",
                      type="int",
                      help=" xxx ",
                      metavar="VALUE",
                      default=4)
    parser.add_option("--bin_size",
                      dest="bin_size",
                      type="int",
                      help="image size reduction",
                      metavar="VALUE",
                      default=9)
    parser.add_option("--defocus",
                      dest="defocus",
                      type="float",
                      help="for defocus-based peak detection threshold, higher defocus, lower threshold",
                      metavar="VALUE",
                      default=1.5)
    parser.add_option("--max_sigma",
                      dest="max_sigma",
                      type="int",
                      help="for peak detection",
                      metavar="VALUE",
                      default=10)
    parser.add_option("--particle_size",
                      dest="particle_size",
                      type="int",
                      help="number of pixels of particles",
                      metavar="VALUE",
                      default=-1)
    (opt, args) = parser.parse_args()
    distance = int(round(opt.particle_size / opt.bin_size))

### scale threshold  between 0.001 and 0.002 based on defocus 
    t = opt.defocus
    tmin = 0.5  ## min defocus 
    tmax = 3.0  ## max defocus 
    rmin = 0.00008 
    rmax = 0.00015
    tnew = (t-tmin)/(tmax-tmin) * (rmax-rmin) + rmin  ## set up real threshold

    # Read input mrc file
    with mrcfile.open(opt.mrc_file, mode='r', permissive=True) as mrc:
        # mrc.header.map = mrcfile.constants.MAP_ID
        header = mrc.header
        body_2d = mrc.data
    n_col = header.nx
    n_row = header.ny
    print("size:", n_col, n_row)

    image_scaled = DataLoader.preprocess_micrograph_local(body_2d, opt.bin_size)
#    image_scaled = median(image_scaled)
#    image_scaled = edge_remove(image_scaled)
#    image_scaled = ndimage.maximum_filter(image_scaled, size = 20)
    print("After binning:", len(image_scaled))
    image_scaled = threshold_local(image_scaled*-1.0,9, method='gaussian', mode='reflect')
    #image_scaled = image_scaled > 1.2 
    #print(image_scaled)
    blobs = blob_doh(image_scaled,
                     min_sigma=3,
                     max_sigma=opt.max_sigma,
                     threshold=tnew,
                     overlap=0.1)

    blobs = blobs.astype(int)
    list_coordinate = local_max(image_scaled, blobs, box_size=distance)

    # Write coordinates to list
    for i in range(len(list_coordinate)):
        # Append mrc_file name to a new column
        list_coordinate[i].append(opt.mrc_file)
        # scale the coordinates to raw images
        list_coordinate[i][0] = list_coordinate[i][0]*opt.bin_size
        list_coordinate[i][1] = list_coordinate[i][1]*opt.bin_size
    write_coordinate(list_coordinate,
                     opt.mrc_file,
                     coordinate_suffix='_local',
                     dirname='local/aligned')


if __name__ == "__main__":
    localPicker()
