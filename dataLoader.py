""""
Script containing dataloader class
"""
## DataLoader for read particles from relion star fileName
## adapted from https://github.com/nejyeah/DeepPicker-python

import os
import mrcfile
import numpy as np
import scipy.ndimage as ndimage
from scipy.misc import imresize
from sklearn import preprocessing
from starReader import starRead

class DataLoader(object):

    @staticmethod
    def bin_2d(body_2d, bin_size):
        # based on the numpy operation to do the bin process
        col = body_2d.shape[0]
        row = body_2d.shape[1]
        scale_col = int(round(col/bin_size))
        scale_row = int(round(row/bin_size))
        patch = np.copy(body_2d[0:scale_col*bin_size, 0:scale_row*bin_size])
        patch_view = patch.reshape(scale_col, bin_size, scale_row, bin_size)
        body_2d_bin = patch_view.mean(axis=3).mean(axis=1)
        return body_2d_bin

    @staticmethod
    def preprocess_particle(particle, model_input_size):
        # resize the particle to fit the model input
        particle = imresize(particle, (model_input_size[1], model_input_size[2]), interp = 'bilinear', mode = 'L')
        # normalize the patch
        mean_value = particle.mean()
        std_value = particle.std()
        particle = (particle - mean_value)/std_value
        return particle

    @staticmethod
    def preprocess_micrograph_local(micrograph, bin_size):
        # Lowpass filter
        micrograph = ndimage.filters.gaussian_filter(micrograph, 0.5)
        # do the bin process
        micrograph = DataLoader.bin_2d(micrograph, bin_size)
        # standarization of  micrographs
        micrograph = preprocessing.scale(micrograph)
        transform = preprocessing.MaxAbsScaler()
        micrograph = transform.fit_transform(micrograph)
        return micrograph

    @staticmethod
    def preprocess_micrograph_picker(micrograph, bin_size):
        # lowpass
        micrograph = ndimage.filters.gaussian_filter(micrograph, 0.1)
        # do the bin process
        micrograph = DataLoader.bin_2d(micrograph, bin_size)
        mean_value = micrograph.mean()
        std_value = micrograph.std()
        micrograph = (micrograph - mean_value) / std_value
        return micrograph

    @staticmethod
    def load_trainData_From_RelionStarFile(starFileName, particle_size,
                                           model_input_size, validation_ratio,
                                           train_number, bin_size):

        particle_array_positive, particle_array_negative = DataLoader.load_Particle_From_starFile(
            starFileName, particle_size, model_input_size, bin_size)
        if train_number < len(particle_array_positive):
            particle_array_positive = particle_array_positive[:train_number,
                                                              ...]
            particle_array_negative = particle_array_negative[:train_number,
                                                              ...]

        np.random.shuffle(particle_array_positive)
        np.random.shuffle(particle_array_negative)

        validation_size = int(validation_ratio*particle_array_positive.shape[0])
        train_size = particle_array_positive.shape[0] - validation_size

        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = np.concatenate(
            (validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = np.concatenate(
            (np.ones(validation_size, dtype=np.int64), np.zeros(validation_size, dtype=np.int64)))

        train_data = particle_array_positive[validation_size:, ...]
        train_data = np.concatenate(
            (train_data, particle_array_negative[validation_size:, ...]))
        train_labels = np.concatenate(
            (np.ones(train_size, dtype=np.int64), np.zeros(train_size, dtype=np.int64)))

        print(train_data.shape, train_data.dtype)
        print(train_labels.shape, train_labels.dtype)
        print(validation_data.shape, validation_data.dtype)
        print(validation_labels.shape, validation_labels.dtype)
        return train_data, train_labels, validation_data, validation_labels

    @staticmethod
    def load_Particle_From_starFile(starFileName,
                                    particle_size,
                                    model_input_size,
                                    bin_size,
                                    produce_negative=True,
                                    negative_distance_ratio=0.4,
                                    negative_number_ratio=1):
        particle_star = starRead(starFileName)
        table_star = particle_star.getByName('data_')
        mrcfilename_list = table_star.getByName('_rlnMicrographName')
        coordinateX_list = table_star.getByName('_rlnCoordinateX')
        coordinateY_list = table_star.getByName('_rlnCoordinateY')

        # creat a dictionary to store the coordinate
        # the key is the mrc file name
        # the value is a list of the coordinates
        coordinate = {}
        #path_star = os.path.split(starFileName)
        for i in range(len(mrcfilename_list)):
            fileName = mrcfilename_list[i]
            #fileName = os.path.join(path_star[0], fileName)
            if fileName in coordinate:
                coordinate[fileName][0].append(int(float(coordinateX_list[i])))
                coordinate[fileName][1].append(int(float(coordinateY_list[i])))
            else:
                coordinate[fileName] = [[], []]
                coordinate[fileName][0].append(int(float(coordinateX_list[i])))
                coordinate[fileName][1].append(int(float(coordinateY_list[i])))
        # read mrc data using the mrcfile library
        particle_array_positive = []
        particle_array_negative = []
        number_total_particle = 0
        for key in coordinate:
            print(key)
            with mrcfile.open(key, mode='r', permissive=True) as mrc:
                header = mrc.header
                body_2d = mrc.data
            n_col = header.nx
            n_row = header.ny
            # show the micrograph with manually picked particles
            # plot the circle of the particle
            # do preprocess to the micrograph
            body_2d = DataLoader.preprocess_micrograph_picker(body_2d, bin_size)
            # bin scale the particle size and the coordinates
            particle_size_bin = int(particle_size / bin_size)
            n_col = int(n_col / bin_size)
            n_row = int(n_row / bin_size)
            for i in range(len(coordinate[key][0])):
                coordinate[key][0][i] = int(coordinate[key][0][i] / bin_size)
                coordinate[key][1][i] = int(coordinate[key][1][i] / bin_size)

            # delete the particle outside the boundry
            radius = int(particle_size_bin / 2)
            i = 0
            while True:
                if i >= len(coordinate[key][0]):
                    break

                coordinate_x = coordinate[key][0][i]
                coordinate_y = coordinate[key][1][i]
                if coordinate_x < radius or coordinate_y < radius or coordinate_y + radius > n_col or coordinate_x + radius > n_row:
                    del coordinate[key][0][i]
                    del coordinate[key][1][i]
                else:
                    i = i + 1

            # number of positive particles
            number_particle = len(coordinate[key][0])
            number_total_particle = number_total_particle + number_particle
            print('number of particles:', number_particle, 'accumulated: ', number_total_particle)

            # extract the positive particles
            # store the particles in a contacted array: particle_array_positive
            for i in range(number_particle):
                coordinate_x = coordinate[key][0][i]
                coordinate_y = coordinate[key][1][i]
                patch = np.copy(
                    body_2d[(coordinate_y - radius):(coordinate_y + radius),
                            (coordinate_x - radius):(coordinate_x + radius)])
                patch = DataLoader.preprocess_particle(patch, model_input_size)
                particle_array_positive.append(patch)
            # extract the negative particles
            # store the particles in a concated array: particle_array_negative
            if produce_negative:
                for i in range(number_particle):
                    while True:
                        isLegal = True
                        coor_x = np.random.randint(radius, n_row - radius)
                        coor_y = np.random.randint(radius, n_col - radius)
                        for j in range(number_particle):
                            coordinate_x = coordinate[key][0][j]
                            coordinate_y = coordinate[key][1][j]
                            distance = ((coor_x - coordinate_x)**2 +
                                        (coor_y - coordinate_y)**2)**0.5
                            patch = np.copy(
                                body_2d[(coor_y - radius):(coor_y + radius),
                                        (coor_x - radius):(coor_x + radius)])
                            # print("radius ", radius, particle_size_bin,patch.shape )
                            if patch.shape == (
                                    radius * 2, radius * 2
                            ) and distance >= negative_distance_ratio * particle_size_bin:
                                patch = DataLoader.preprocess_particle(
                                    patch, model_input_size)
                                particle_array_negative.append(patch)
                                isLegal = False
                                break
                        if isLegal is False:
                            break

        if produce_negative:
            #print("total: ", np.array(particle_array_positive[0]).shape)
            particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle,model_input_size[1],model_input_size[2], 1)
            particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle,model_input_size[1],model_input_size[2], 1)
            return particle_array_positive, particle_array_negative

        else:
            particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle,model_input_size[1],model_input_size[2], 1)
            return particle_array_positive
