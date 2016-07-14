import numpy as np, keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import math, sys
#from matplotlib import pyplot as plt
from random import randint
#plt.switch_backend('Qt4Agg')
import keras

def get_setpoints_traj(run_time):
    import math
    desired_setpoints = []
    for i in range(run_time):
        turbine = math.sin(i*1.0/7.0)
        if turbine < 0: turbine = 0
        turbine *= 0.1
        turbine += 0.7
        fuel_cell = 0.75
        desired_setpoints.append([turbine, fuel_cell])
    return desired_setpoints

def init_model(num_hnodes = 35):
    model = Sequential()
    model.add(Dense(num_hnodes, input_dim=23, init='he_uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(2,init='he_uniform'))
    model.compile(loss='mean_absolute_error', optimizer='Nadam')
    return model

def data_preprocess(filename = 'ColdAir.csv', downsample_rate=25):

     #Import training data and clear away the two top lines
    train_data = np.loadtxt(filename, delimiter=',', skiprows= 2 )

    #Splice data (downsample)
    ignore = np.copy(train_data)
    train_data = train_data[0::downsample_rate]
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            if ( i != train_data.shape[0]-1):
                train_data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,j].sum()/downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                train_data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue,j].sum()/residue

    #Normalize between 0-0.99
    normalizer = np.zeros(train_data.shape[1], dtype=np.float64)
    min = np.zeros(len(train_data[0]), dtype=np.float64)
    max = np.zeros(len(train_data[0]), dtype=np.float64)
    for i in range(len(train_data[0])):
        min[i] = np.amin(train_data[:,i])
        max[i] = np.amax(train_data[:,i])
        normalizer[i] = max[i]-min[i] + 0.00001
        train_data[:,i] = (train_data[:,i] - min[i])/ normalizer[i]

    return train_data, max, min

def novelty(weak_matrix, archive, k = 10):
    import bottleneck
    #Handle early gens with archive size less that 10
    if (len(archive) < k):
        k = len(archive)

    novel_matrix = np.zeros(len(archive))
    for i in range(len(archive)):
        novel_matrix[i] = np.sum(np.square(weak_matrix - archive[i]))

    #k-nearest neighbour algorithm
    k_neigh = bottleneck.partsort(novel_matrix, k)[:k] #Returns a subarray of k smallest novelty scores

    #Return novelty score as the average Euclidean distance (behavior space) between its k-nearest neighbours
    return np.sum(k_neigh)/k

def import_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    from keras.models import model_from_json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch

def print_results(setpoints, initial_state, simulator, model_name):
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')
    model = import_arch()
    model.load_weights(model_name)

    indices = [11,15]
    tr_output = np.zeros(len(indices) * (len(setpoints) -1))
    tr_output = np.reshape(tr_output, (len(indices), len(setpoints)-1))

    weakness = np.zeros(len(indices))
    input = np.append(initial_state, setpoints[0])
    input = np.reshape(input, (1,23))

    for example in range(len(setpoints)-1):#For all training examples
        #Get the controller output and fill in the controls
        control_out = model.predict(input)
        input[0][19] = control_out[0][0]
        input[0][20] = control_out[0][1]

        #Use the simulator to get the next state
        model_out = simulator.predict(np.reshape(input[0][0:21], (1,21))) #Time domain simulation
        #Calculate error (weakness)
        for index in range(len(indices)):
            #weakness[index] += math.fabs(model_out[0][indices[index]] - setpoints[example][index])#Time variant simulation
            tr_output[index][example] = model_out[0][indices[index]]

        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]

        #Fill in next setpoints
        input[0][21] = setpoints[example+1][0]
        input[0][22] = setpoints[example+1][1]


    #ignore = abs(track_time_target - track_time_output)
    setpoints = np.array(setpoints)
    setpoints = np.transpose(setpoints)

    #plt.plot(ignore[index], 'r--',label='Target Index: ' + str(index))
    axes = plt.gca()
    axes.set_ylim([0.5, 0.9])
    plt.plot(tr_output[0], 'r--',label='Turbine Output')
    plt.plot(setpoints[0], 'b-',label='Turbine Setpoint ')
    plt.plot(tr_output[1], 'g--', label='Fuel Cell Output')
    plt.plot(setpoints[1], 'y-', label='Fuel Cell Setpoint' )
    plt.legend( loc='upper right',prop={'size':12})
    #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    plt.show()


    # for index in range(len(indices)):
    #     #plt.plot(ignore[index], 'r--',label='Target Index: ' + str(index))
    #     plt.plot(tr_output[index], 'r--',label='Controller Output: ' + str(index))
    #     plt.plot(setpoints[index], 'b-',label='Setpoint: ' + str(index))
    #     plt.legend( loc='upper right',prop={'size':10})
    #     #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    #     plt.show()

def metrics(train_data, model, max, min, time_domain = True, filename = 'ColdAir.csv', downsample_rate=25, noise = False, noise_mag = 0.1):
    #Load model
    #model = theanets.Regressor.load(model_name)
    input = np.reshape(train_data[0], (1, 21)) #First input to the simulatior

    #Track across time steps
    track_time_target = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))
    track_time_output = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))
    error = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))

    for example in range(len(train_data)-1):#For all training examples
        if (time_domain):
            model_out = model.predict(input) #Time domaain simulation
        else:
             model_out = model.predict(np.reshape(train_data[example], (1, 21))) #Non-domain simulation
        #Track index
        for index in range(19):
            track_time_output[index][example] = model_out[0][index] * (max[index]-min[index] + 0.00001) + min[index]
            track_time_target[index][example] = train_data[example+1][index] * (max[index]-min[index] + 0.00001) + min[index]
        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][19] = train_data[example+1][19]
        input[0][20] = train_data[example+1][20]
    error = abs(track_time_output - track_time_target)
    for index in range(19):
        error[index][:] = error[index][:] * 100 / max[index]
    return error

def return_results(time_domain = True, model_name = 'Evolutionary/temp/0',filename = 'ColdAir.csv', downsample_rate=25):
    #Load model
    import theanets
    model = theanets.Regressor.load(model_name)

    #Import training data and clear away the two top lines
    train_data = np.loadtxt(filename, delimiter=',', skiprows= 2 )

    #Splice data (downsample)
    ignore = np.copy(train_data)
    train_data = train_data[0::downsample_rate]
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            if ( i != train_data.shape[0]-1):
                train_data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,j].sum()/downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                train_data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue,j].sum()/residue


    #Normalize between 0-0.99
    normalizer = np.zeros(len(train_data[0]), dtype=np.float64)
    min = np.zeros(len(train_data[0]), dtype=np.float64)
    max = np.zeros(len(train_data[0]), dtype=np.float64)
    for i in range(len(train_data[0])):
        min[i] = np.amin(train_data[:,i])
        max[i] = np.amax(train_data[:,i])
        normalizer[i] = max[i]-min[i] + 0.00001
        train_data[:,i] = (train_data[:,i] - min[i])/ normalizer[i]

    ##EO FILE IO##

    print ('TESTING NOW')
    input = np.reshape(train_data[0], (1, 21)) #First input to the simulatior
    error = np.zeros(21) #Error array to track error for each variables

    #Track across time steps
    track_time_target = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))
    track_time_output = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))

    for example in range(len(train_data)-1):#For all training examples

        if (time_domain):
            model_out = model.predict(input) #Time domaain simulation
        else:
             model_out = model.predict(np.reshape(train_data[example], (1, 21))) #Non-domain simulation

        #Track index
        for index in range(19):
            track_time_output[index][example] = model_out[0][index] * normalizer[index] + min[index]
            track_time_target[index][example] = train_data[example+1][index] * normalizer[index] + min[index]

        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][19] = train_data[example+1][19]
        input[0][20] = train_data[example+1][20]


    list = []
    list.append(track_time_target)
    list.append(track_time_output)
    return list

    # #ignore = abs(track_time_target - track_time_output)
    # for index in range(19):
    #     #plt.plot(ignore[index], 'r--',label='Target Index: ' + str(index))
    #     plt.plot(track_time_target[index], 'r--',label='Target Index: ' + str(index))
    #     plt.plot(track_time_output[index], 'b-',label='Output Index: ' + str(index))
    #     plt.legend( loc='upper right' )
    #     plt.savefig('Results/' + 'Index' + str(index) + '.png')
    #     plt.show()

def mutate(model_in, model_out, many_strength = 1, much_strength = 1):
    #NOTE: Takes in_num file, mutates it and saves as out_num file, many_strength denotes how many mutation while
    # much_strength controls how strong each mutation is

    w = model_in.get_weights()
    for many in range(many_strength):#Number of mutations
        i = randint(0, len(w)-1)
        if len(w[i].shape) == 1: #Bias
            j = randint(0, len(w[i])-1)
            w[i][j] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
            # if (randint(1, 100) == 5): #SUPER MUTATE
            #     w[i][j] += np.random.normal(-1 * much_strength, 1 * much_strength)
        else:  # Bias
            j = randint(0, len(w[i]) - 1)
            k = randint(0, len(w[i][j]) - 1)
            w[i][j][k] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
            # if (randint(1, 100) == 5):  # SUPER MUTATE
            #     w[i][j][k] += np.random.normal(-1 * much_strength, 1 * much_strength)

    model_out.set_weights(w) #Save weights

def rec_weakness(setpoints, initial_state, model, n_prev=7, novelty = False, test = False): #Calculates weakness (anti fitness) of RECCURRENT models
    weakness = np.zeros(19)
    input = np.reshape(train_data[0:n_prev], (1, n_prev, 21))  #First training example in its entirety

    for example in range(len(train_data)-n_prev):#For all training examples
        model_out = model.predict(input) #Time domain simulation
        #Calculate error (weakness)
        for index in range(19):
            weakness[index] += math.fabs(model_out[0][index] - train_data[example+n_prev][index])#Time variant simulation
        #Fill in new input data
        for k in range(len(model_out)): #Modify the last slot
            input[0][0][k] = model_out[0][k]
            input[0][0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][0][19] = train_data[example+n_prev][19]
        input[0][0][20] = train_data[example+n_prev][20]
        input = np.roll(input, -1, axis=1)  # Track back everything one step and move last one to the last row
    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(train_data)-n_prev)
    else:
        return np.sum(np.square(weakness))

def ff_weakness(setpoints, initial_state, simulator, model, novelty = False, test = False): #Calculates weakness (anti fitness) of FEED-FORWARD models

    indices = [11,15]
    weakness = np.zeros(len(indices))
    input = np.append(initial_state, setpoints[0])
    input = np.reshape(input, (1,23))

    for example in range(len(setpoints)-1):#For all training examples
        #Get the controller output and fill in the controls
        control_out = model.predict(input)
        input[0][19] = control_out[0][0]
        input[0][20] = control_out[0][1]

        #Use the simulator to get the next state
        model_out = simulator.predict(np.reshape(input[0][0:21], (1,21))) #Time domain simulation
        #Calculate error (weakness)
        for index in range(len(indices)):
            weakness[index] += math.fabs(model_out[0][indices[index]] - setpoints[example][index])#Time variant simulation

        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]

        #Fill in next setpoints
        input[0][21] = setpoints[example+1][0]
        input[0][22] = setpoints[example+1][1]

    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(setpoints)-1)
    else:
        return np.sum(np.square(weakness))