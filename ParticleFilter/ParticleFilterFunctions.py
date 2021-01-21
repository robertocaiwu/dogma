# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 09:40:51 2017

Author: Masha Itkina
Collaborators: Henry Shi and Michael Anderson

Main function for the particle filter algorithm implementation as outlined in:

D. Nuss, S. Reuter, M. Thom, T. Yuan, G. Krehl, M. Maile, A. Gern, and K. Dietmayer. A
random finite set approach for dynamic occupancy grid maps with real-time application. arXiv,
abs/1605.02406, 2016.

The simulation example considers a stationary ego vehicle and 2 vehicles traveling at 10 m/s in opposite directions.
The grids have 33 cm resolution.

"""
import cv2
import colorsys
from PlotTools import colorwheel_plot, particle_plot
import math
import os
import sys
import time
import hickle as hkl
import pickle
from Simulator import *
from Grid import *
from Particle import *
from Resample import *
from StatisticMoments import *
from PersistentParticleUpdate import *
from MassUpdate import *
from OccupancyPredictionUpdate import *
from ParticleAssignment import *
from ParticlePrediction import *
from NewParticleInitialization import *
from scipy.stats import itemfreq
import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')


seed = 1987
np.random.seed(seed)


sys.path.insert(0, '..')

DATA_DIR = "../data/sensor_grids/"
OUTPUT_DIR = "../data/dogma/"
MEAS_DIR = "../data/meas_grids/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def crop_center(img, crop):
    m, x, y = img.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    return img[:, starty:starty+crop, startx:startx+crop]

# Populate the Dempster-Shafer measurement masses.


def create_DST_grids(grids, meas_mass=0.95):

    data = []

    for i in range(grids.shape[0]):

        grid = grids[i, :, :]
        free_array = np.zeros(grid.shape)
        occ_array = np.zeros(grid.shape)

        # occupied indices
        indices = np.where(grid == 1)
        occ_array[indices] = meas_mass

        # free indices
        indices = np.where(grid == 2)
        free_array[indices] = meas_mass

        # car
        indices = np.where(grid == 3)
        occ_array[indices] = 1.
        free_array[indices] = 0.

        data.append(np.stack((free_array, occ_array)))

    data = np.array(data)

    return data


def create_grids():

    data = []

    for path in sorted(os.listdir(MEAS_DIR)):

        im = cv2.imread(os.path.join(MEAS_DIR, path))

        occ = im[:, :, 0]
        free = im[:, :, 1]

        free = np.divide(free, 255.0)
        occ = np.divide(occ, 255.0)

        data.append(np.stack((free, occ)))

    data = np.array(data)

    return data


def particle_filter_functions():

    with open(os.path.join(DATA_DIR, 'simulation.pickle'), 'rb') as f:
        start = time.time()

        # load sensor grid data (list of arrays)
        simulation_data = pickle.load(f,encoding='bytes')
        print(simulation_data)
        [grids, global_x_grid, global_y_grid] = simulation_data

        # convert to numpy array
        grids = np.array(grids)

        end = time.time()
        # print("Loading simulation datatook", end - start, len(grids), grids[0].shape

    # crop grids to the desired shape
    shape = (128, 128)
    grids = np.array(grids)
    grids = crop_center(grids, shape[0])
    # print(grids.shape

    do_plot = True  # Toggle me for DOGMA plots!

    # PARAMETERS
    p_B = 0.02                                            # birth probability
    # number of new born particles
    Vb = 2*10**4
    # number of consistent particles
    V = 2*10**5
    state_size = 4                                        # number of states: p,v: 4
    # information ageing (discount factor)
    alpha = 0.9

    # association probability: only relevant for Doppler measurements
    p_A = 1.0
    # measurement frequency (10 Hz)
    T = 0.1
    # particle persistence probability
    p_S = 0.99
    res = 1.                                              # resolution of the grid cells

    # velocity, acceleration variance initialization
    scale_vel = 12.
    scale_acc = 2.

    # position, velocity, acceleration process noise
    process_pos = 0.06
    process_vel = 2.4
    process_acc = 0.2

    # # print(debug values
    verbose = False

    # for plotting thresholds
    mS = 3.
    epsilon = 10.
    epsilon_occ = 0.75

    # index where PF was interrupted
    index_stopped = 0

    # initialize a grid
    start = time.time()
    grid_cell_array = GridCellArray(shape, p_A)
    end = time.time()
    # print("grid_cell_array initialization took", end - start

    # initialize a particle array
    start = time.time()
    particle_array = ParticleArray(V, grid_cell_array.get_shape(
    ), state_size, T, p_S, scale_vel, scale_acc, process_pos, process_vel, process_acc)
    end = time.time()
    # print("particle_array initialization took", end - start

    # data: [N x 2 x W x D]
    # second dimension is masses {0: m_free, 1: m_occ}
    # in original grid: 0: unknown, 1: occupied, 2: free (raw data)
#	data = create_DST_grids(grids)
    data = create_grids()

    # number of measurements in the run
    N = data.shape[0]

    # list of 4x128x128 grids with position, velocity information
    DOGMA = []
    var_x_vel = []
    var_y_vel = []
    covar_xy_vel = []
    var_x_acc = []
    var_y_acc = []
    covar_xy_acc = []

    # run particle filter iterations
    for i in range(N):

        start = time.time()

        # initializes a measurement cell array
        meas_free = data[i, 0, :, :].flatten()
        meas_occ = data[i, 1, :, :].flatten()

        meas_cell_array = MeasCellArray(
            meas_free, meas_occ, grid_cell_array.get_shape(), pseudoG=1.)

        # algorithm 1: ParticlePrediction (stored in particle_array)
        ParticlePrediction(particle_array, grid_cell_array, res=res)

        # algorithm 2: ParticleAssignment (stored in particle_array)
        ParticleAssignment(particle_array, grid_cell_array)

        # algorithm 3: OccupancyPredictionUpdate (stored in grid_cell_array)
        OccupancyPredictionUpdate(
            meas_cell_array, grid_cell_array, particle_array, p_B, alpha, check_values=verbose)
        #MassUpdate(meas_cell_array, grid_cell_array, p_B, alpha, check_values = verbose)

        # algorithm 4: PersistentParticleUpdate (stored in particle_array)
        PersistentParticleUpdate(
            particle_array, grid_cell_array, meas_cell_array, check_values=verbose)

        # algorithm 5: NewParticleInitialization
        if p_B == 0:
            empty_array = True
        else:
            empty_array = False
        birth_particle_array = ParticleArray(Vb, grid_cell_array.get_shape(
        ), state_size, T, p_S, scale_vel, scale_acc, process_pos, process_vel, process_acc, birth=True, empty_array=empty_array)
        NewParticleInitialization(
            Vb, grid_cell_array, meas_cell_array, birth_particle_array, check_values=verbose)

        # algorithm 6: StatisticMoments (stored in grid_cell_array)
        StatisticMoments(particle_array, grid_cell_array)

    #	if (i + 1) > index_stopped:

    #		newDOGMA, new_var_x_vel, new_var_y_vel, new_covar_xy_vel = get_dogma(grid_cell_array, grids, state_size, grids[i,:,:], shape)

#		var_x_vel.append(new_var_x_vel)
#		var_y_vel.append(new_var_y_vel)
#		covar_xy_vel.append(new_covar_xy_vel)

        # save the DOGMA at this timestep: before we had occupancy, free, but this is actually not the real occupancy plot
        # so we will just use the measurement grid for now
#		if (i+1) > index_stopped:
#			DOGMA.append(newDOGMA)
#			print("really?")

        # algorithm 7: Resample
        # skips particle initialization for particle_array_next because all particles will be copied in
        particle_array_next = ParticleArray(V, grid_cell_array.get_shape(), state_size, T, p_S,
                                            scale_vel, scale_acc, process_pos, process_vel, process_acc, empty_array=True)
        Resample(particle_array, birth_particle_array,
                 particle_array_next, check_values=verbose)

        # switch to new particle array
        particle_array = particle_array_next
        particle_array_next = None

        end = time.time()
        # print("### Iteration took: ", end - start

        # Plotting: The environment is stored in grids[i] (matrix of  values (0,1,2))
        #           The DOGMA is stored in DOGMA[i]
#		if (do_plot):
#			head_grid = dogma2head_grid(DOGMA[i], var_x_vel[i], var_y_vel[i], covar_xy_vel[i], mS, epsilon, epsilon_occ)
#			occ_grid = grids[i,:,:]
#			title = "DOGMa Iteration %d" % i
#			colorwheel_plot(head_grid, occ_grid=occ_grid, m_occ_grid = DOGMA[i][0,:,:], title=os.path.join(OUTPUT_DIR, title), show=True, save=True)

        # print("Iteration ", i, " complete"
        # print("### Saving result"
        # print("#####################"

        gm_img = np.zeros((shape[1], shape[0], 3), np.uint8)

        for y in range(shape[1]):
            for x in range(shape[0]):
                ind = y * shape[0] + x
                c = grid_cell_array.cells[ind]
                occ = c.m_occ + 0.5 * (1.0 - c.m_occ - c.m_free)
                temp = int(occ * 255)

                covar = np.array([[c.var_x_vel, c.covar_xy_vel], [
                                 c.covar_xy_vel, c.var_y_vel]])
                if abs(np.linalg.det(covar)) < 10**(-6):
                    mdist = 0.
                else:
                    mdist = np.array([c.mean_x_vel, c.mean_y_vel]).dot(
                        np.linalg.inv(covar)).dot(np.array([c.mean_x_vel, c.mean_y_vel]).T)

                if occ >= 0.6 and mdist >= 4:
                    # print("bims")
                    #print("hello >= 0.6")
                    angle = math.atan2(c.mean_y_vel, c.mean_x_vel)
                    angle = math.degrees(angle)

                    r, g, b = colorsys.hsv_to_rgb(angle / 360.0, 1.0, 1.0)

                    gm_img[int(y), int(x), 0] = int(b * 255)
                    gm_img[int(y), int(x), 1] = int(g * 255)
                    gm_img[int(y), int(x), 2] = int(r * 255)
                else:
                    gm_img[int(y), int(x), 0] = 255 - temp
                    gm_img[int(y), int(x), 1] = 255 - temp
                    gm_img[int(y), int(x), 2] = 255 - temp

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'dogm_gm%d.png' % i), gm_img)

        parts_img = np.zeros((shape[1], shape[0], 3), np.uint8)

        for part in particle_array:
            x = part[0]
            y = part[1]

            if (x >= 0 and x < shape[0]) and (y >= 0 and y < shape[1]):
                parts_img[int(x), int(y), 0] = 0
                parts_img[int(x), int(y), 1] = 0
                parts_img[int(x), int(y), 2] = 255

        cv2.imwrite(os.path.join(
            OUTPUT_DIR, 'dogm_particles%d.png' % i), parts_img)

        meas_img = np.zeros((shape[1], shape[0], 3), np.uint8)

        for y in range(shape[1]):
            for x in range(shape[0]):
                ind = y * shape[0] + x
                c = meas_cell_array.cells[ind]
                occ = c.m_occ + 0.5 * (1.0 - c.m_occ - c.m_free)
                temp = int(occ * 255)

                meas_img[int(y), int(x), 0] = int(c.m_occ * 255)
                meas_img[int(y), int(x), 1] = int(c.m_free * 255)
                meas_img[int(y), int(x), 2] = 0

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'meas_grid%d.png' % i), meas_img)


#		hkl.dump([DOGMA, var_x_vel, var_y_vel, covar_xy_vel], os.path.join(OUTPUT_DIR, 'DOGMA.hkl'), mode='w')
#		# print("DOGMA written to hickle file."

    return

# Save DOGMa: 4x128x128 (occupied mass, free mass, velocity x, velocity y, original measurement grid)
# and DOGMa statistics: velocity variances and covariances


def get_dogma(grid_cell_array, grids, state_size, meas_grid, shape):

    ncells = grid_cell_array.get_length()

    if state_size == 4:
        posO = np.zeros([ncells])
        posF = np.zeros([ncells])
        velX = np.zeros([ncells])
        velY = np.zeros([ncells])
        var_x_vel = np.zeros([ncells])
        var_y_vel = np.zeros([ncells])
        covar_xy_vel = np.zeros([ncells])

        for i in range(ncells):
            posO[i] = grid_cell_array.get_cell_attr(i, "m_occ")
            posF[i] = grid_cell_array.get_cell_attr(i, "m_free")
            velX[i] = grid_cell_array.get_cell_attr(i, "mean_x_vel")
            velY[i] = grid_cell_array.get_cell_attr(i, "mean_y_vel")
            var_x_vel[i] = grid_cell_array.get_cell_attr(i, "var_x_vel")
            var_y_vel[i] = grid_cell_array.get_cell_attr(i, "var_y_vel")
            covar_xy_vel[i] = grid_cell_array.get_cell_attr(i, "covar_xy_vel")

        posO = posO.reshape(shape)
        posF = posF.reshape(shape)
        velX = velX.reshape(shape)
        velY = velY.reshape(shape)
        var_x_vel = var_x_vel.reshape(shape)
        var_y_vel = var_y_vel.reshape(shape)
        covar_xy_vel = covar_xy_vel.reshape(shape)

        newDOGMA = np.stack((posO, posF, velX, velY, meas_grid))

        return newDOGMA, var_x_vel, var_y_vel, covar_xy_vel

    else:
        raise Exception("Unexpected state size.")
        return


def dogma2head_grid(dogma, var_x_vel, var_y_vel, covar_xy_vel, mS=4., epsilon=0.5, epsilon_occ=0.1):
    """Create heading grid for plotting tools from a DOGMA.
    USAGE:
            head_grid = dogma2head_grid(dogma, (epsilon) )
    INPUTS:
            dogma - (np.ndarray) Single DOGMA tensor (supports size of 4 or 6)
            epsilon - (opt)(float) Minimum cell vel mag required to plot heading
    OUTPUTS:
            head_grid - (np.matrix) Grid (of same shape as each vel grid) containing
                                    object headings at each cell, in rad
    """
    grid_shape = dogma[0, :, :].shape

    # Initialize grid with None's; this distinguishes from a 0rad heading!
    head_grid = np.full(grid_shape, None, dtype=float)
    vel_x, vel_y = dogma[2:4, :, :]
    m_occ, m_free = dogma[0:2, :, :]
    meas_grid = dogma[4, :, :]

    # Fill grid with heading angles where we actually have velocity
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):

            # mahalanobis distance
            covar = np.array([[var_x_vel[i, j], covar_xy_vel[i, j]], [
                             covar_xy_vel[i, j], var_y_vel[i, j]]])
            if abs(np.linalg.det(covar)) < 10**(-6):
                mdist = 0.
            else:
                mdist = np.array([vel_x[i, j], vel_y[i, j]]).dot(
                    np.linalg.inv(covar)).dot(np.array([vel_x[i, j], vel_y[i, j]]).T)

            mag = np.sqrt(vel_x[i, j]**2 + vel_y[i, j]**2)

            # occupied and with velocity
            if ((mdist > mS) and (m_occ[i, j] > epsilon_occ)):
                heading = np.arctan2(vel_y[i, j], vel_x[i, j])
                head_grid[i, j] = heading

    return head_grid


if __name__ == "__main__":
    particle_filter_functions()
