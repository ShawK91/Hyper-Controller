import numpy as np, re, theanets
from random import randint
import modules_controller as mod
import bottleneck
from keras.models import model_from_json



#MACRO
import_new = True   #Set to true if import new seed from Model/Singular
new_popn = True #Set to true if initializing a new population from Evolutionary/0
seed_size = 100 #Size of initial seed population
freeloaders = 5 #Quantity of popn selected despite being infeasible
n_prev = 7
total_gen = 25000
is_recurrent = False





#Method to save/copy models
def import_models(in_filename, save_filename, model_arch):
    model_arch.load_weights(in_filename)
    model_arch.save_weights(save_filename, overwrite=True)
    model_arch.compile(loss='mae', optimizer='adam')

def save_models(nn_popn, popn_size):
    for i in range(popn_size):
        nn_popn[i].save_weights('Evolutionary/' + str(i), overwrite=True)

def get_model_arch(import_new, seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    if import_new:
        model_arch = mod.init_model(15)
        json_string = model_arch.to_json()
        with open(seed, 'w') as outfile:
            json.dump(json_string, outfile)
    else:
        with open(seed) as json_file:
            json_data = json.load(json_file)
        model_arch = model_from_json(json_data)
    return model_arch



def dataio(split = 1000):
    data, max, min = mod.data_preprocess()#Process data and populate global variable train_data
    return data[0:1]
    train_data = data[0:split]
    valid_data = data[split:len(data)]
    return train_data, valid_data





def main():
    run_time = 200
    popn_size = 2 * seed_size
    ##### DATA IO ######
    setpoints = mod.get_setpoints_traj(run_time)
    #Get simulator
    simulator = mod.import_arch('Evolutionary/sim.json')
    simulator.load_weights('Evolutionary/sim.h5')
    simulator.compile(loss='mae', optimizer='adam')


    model_arch = get_model_arch(import_new) #Get model architecture
    model_arch.save_weights('Evolutionary/0', overwrite=True)
    initial_state = dataio() #Import initial state
    ### DECLARE A 3-tuple LIST TO HOLD POPULATION, FITNESS & NOVELTY respectively
    population = np.zeros(popn_size * 2, dtype=np.float64)
    population = np.reshape(population, (popn_size, 2))
    for x in range(popn_size): # Naming our models
        population[x][0] = x

    # Form the neural network correspnding to population (nn_popn)
    if (new_popn):  # Form new population
        nn_popn = []
        if import_new:
            for i in range(popn_size):
                nn_popn.append(get_model_arch(import_new))
                nn_popn[i].compile(loss='mae', optimizer='adam')

        for i in range(popn_size - 1):
            mod.mutate(nn_popn[0], nn_popn[i + 1], 10, 10)

        #Invoke generation, archive and metrics
        weakness_tracker = np.zeros(1)#MSE tracker
        generation = 0
        archive = np.empty([1,2])
        if is_recurrent:
            seed_archive = mod.rec_weakness(setpoints, initial_state, simulator, nn_popn[0], n_prev, True)
        else:
            seed_archive = mod.ff_weakness(setpoints, initial_state, simulator, nn_popn[0], True)
        for index in range (2):
            archive[0][index] = seed_archive[index]

    else:

        nn_popn = []
        for i in range(popn_size):
            nn_popn.append(model_arch)
            nn_popn[i].load_weights('Evolutionary/' + str(i))
            model_arch.compile(loss='mae', optimizer='adam')

        #Load generation and weakness tracker
        weakness_tracker = np.loadtxt('Evolutionary/Files/weakness.csv', delimiter=',' )
        generation = int(np.loadtxt('Evolutionary/Files/generation.csv', delimiter=',' ))


    ################# LOOP ######################
    best_weakness = 1000000000 #Magic large constraint to start with (IGNORE)
    gen_since = 0 #Generation since last progress


    gen_array = np.empty(1)
    restart_assess = True
    if (new_popn or import_new):
        restart_assess = False

    while generation < total_gen:
        generation = generation + 1

        #NATURAL SELECTION
        for x in range(popn_size): #Evaluate weakness and novelty for each
            if ((generation == 1 or x > (seed_size - 1)) or restart_assess): #For first time
                if is_recurrent:
                    population[x][1] = mod.rec_weakness(setpoints, initial_state, simulator,nn_popn[int(population[x][0])], n_prev)
                else:
                    population[x][1] = mod.ff_weakness(setpoints, initial_state, simulator,nn_popn[int(population[x][0])])
        restart_assess = False
        population = population[population[:, 1].argsort()]  ##Ranked on fitness (reverse of weakness) s.t. 0 index is the best


        #Choose luck portion of new popn based (Freeloader phase)
        chosen = np.zeros(freeloaders)
        for x in range(freeloaders):
            lucky = randint(seed_size - freeloaders + x, popn_size - 1)
            while (lucky in chosen):
                lucky = randint(seed_size - freeloaders + x, popn_size - 1)
            chosen[x] = lucky
            population[seed_size - freeloaders + x][0], population[lucky][0]  = population[lucky][0], population[seed_size - freeloaders + x][0]

        #Mutate to renew population
        for x in range(seed_size):
            many = 1
            much = randint(1,5)
            if (randint(1,100) == 91):
                many = randint(1,10)
                much = randint(1,100)
            mod.mutate(nn_popn[int(population[x][0])], nn_popn[int(population[x+seed_size][0])], many, much)

        ########### End of core evol. algorithm #######


        #Method to save models and weakness
        if (generation % 100 == 0):
            save_models(nn_popn, popn_size)
            #Generation
            gen_array[0] = generation
            np.savetxt('Evolutionary/Files/trad_generation.csv', gen_array, delimiter=',')

        ##UI
        if (population[0][1] < best_weakness):
            gen_since = 0
            best_weakness = population[0][1]
        else:
            gen_since += 1

        if (generation % 10 == 0):


            print 'Gen:', generation, 'Best:', '%.2f' % best_weakness, 'Fails', gen_since
            weakness_tracker = np.append(weakness_tracker, best_weakness)
            np.savetxt('Evolutionary/Files/trad_weakness.csv', weakness_tracker, delimiter=',')





if __name__ == '__main__':
    main()
