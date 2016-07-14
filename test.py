import numpy as np, sys, copy
from random import randint
import modules_controller as mod
from keras.models import model_from_json

#MACRO
import_new = True   #Set to true if import new seed from Model/Singular
new_popn = True #Set to true if initializing a new population from Evolutionary/0
seed_size = 100 #Size of initial seed population
freeloaders = 5 #Freeloaders are weak candidates that are kept in popn by pure luck (freeloaders = 0 --> elitism)
total_gen = 25000
is_recurrent = False
n_prev = 7



#Method to save/copy models
def import_models(in_filename, save_filename, model_arch):
    model_arch.load_weights(in_filename)
    model_arch.save_weights(save_filename, overwrite=True)
    model_arch.compile(loss='mae', optimizer='adam')

def save_models(nn_popn, popn_size):
    for i in range(popn_size):
        nn_popn[i].save_weights('Evolutionary/' + str(i), overwrite=True)

def get_model_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch



def dataio(split = 1000):
    data, max, min = mod.data_preprocess()#Process data and populate global variable train_data
    train_data = data[0:split]
    valid_data = data[split:len(data)]
    return train_data, valid_data

def main():

    # model = get_model_arch()
    # model.load_weights('Evolutionary/seed.h5')
    # train_data, valid_data = dataio()
    # weak = mod.ff_weakness(train_data, model)
    # print (weak / len(train_data))
    #
    # sys.exit()

    popn_size = 2 * seed_size
    ##### DATA IO ######
    train_data, valid_data = dataio()
    model_arch = get_model_arch() #Get model architecture
    model_arch.load_weights('Evolutionary/seed.h5')

    ### DECLARE A 2-tuple LIST TO HOLD POPULATION AND FITNESS such that index 0 --> model_name and index 1 --> fitness
    population = np.zeros(seed_size * 4, dtype=np.float64)
    population = np.reshape(population, (seed_size * 2, 2))
    for x in range(popn_size): # Make sure the initial mutation happens JUST A MAGIC SHORTCUT
        population[x][0] = x

    #Form the neural network correspnding to population (nn_popn)
    if (new_popn): #Form new population
        if (import_new):#Import model
            import_models('Evolutionary/seed.h5', 'Evolutionary/0', model_arch)

        nn_popn = []
        for i in range(popn_size):
            nn_popn.append(get_model_arch())
            nn_popn[i].load_weights('Evolutionary/0')
            nn_popn[i].compile(loss='mae', optimizer='adam')

        for i in range(popn_size-1):
            mod.mutate(nn_popn[0], nn_popn[i+1], 10, 10)

    for i in range(popn_size):
        weak = mod.ff_weakness(train_data, nn_popn[i])
        print (weak / len(train_data))

        #Invoke generation and metrics
        weakness_tracker = np.zeros(1)#MSE tracker
        track_valid = np.zeros(1)#Metrics tracker
        generation = 0

    else:
        nn_popn = []
        for i in range(popn_size):
            nn_popn.append(model_arch)
            nn_popn[i].load_weights('Evolutionary/' + str(i))
            model_arch.compile(loss='mae', optimizer='adam')

        #Load generation and metrics
        track_valid = np.loadtxt('Evolutionary/Files/valid.csv', delimiter=',' )
        weakness_tracker = np.loadtxt('Evolutionary/Files/trad_weakness.csv', delimiter=',' )
        generation = int(np.loadtxt('Evolutionary/Files/trad_generation.csv', delimiter=',' ))


    sys.exit()
    #Loop starts
    best_weakness = 1000000000 #Magic super large number to beat
    gen_since = 0 #Generation since last progress

    gen_array = np.empty(1)
    restart_assess = True
    if (new_popn or import_new):
        restart_assess = False

    while (generation < total_gen): ############### START LOOP ###################################
        print generation
        generation += 1
        #NATURAL SELECTION
        for x in range(popn_size): #Evaluate weakness
            if ((generation == 1 or x > (seed_size-1)) or restart_assess): #For first time assign to all otherwise only assign to x > seedsize
                if is_recurrent:
                    population[x][1] = mod.rec_weakness(train_data, nn_popn[int(population[x][0])], n_prev)  # Assign weakness
                else:
                    population[x][1] = mod.ff_weakness(train_data, nn_popn[int(population[x][0])])#Assign weakness
        restart_assess = False
        population = population[population[:,1].argsort()]##Ranked on fitness (reverse of weakness) s.t. 0 index is the best

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
        if (population[0][1]/len(train_data) < best_weakness):
            gen_since = 0
            best_weakness = population[0][1]/len(train_data)
        else:
            gen_since += 1

        if (generation % 10 == 0):
            if is_recurrent:
                valid_score = mod.rec_weakness(valid_data, nn_popn[int(population[0][0])],n_prev) / len(valid_data)
            else:
                valid_score = mod.ff_weakness(valid_data, nn_popn[int(population[0][0])]) / len(valid_data)

            print 'Gen:', generation, 'Best:', '%.2f' % best_weakness, 'Valid: ', '%.2f' % valid_score, 'Fails', gen_since
            weakness_tracker = np.append(weakness_tracker, best_weakness)
            np.savetxt('Evolutionary/Files/trad_weakness.csv', weakness_tracker, delimiter=',')
            track_valid = np.append(track_valid, valid_score)
            np.savetxt('Evolutionary/Files/valid.csv', track_valid, delimiter=',')

    ########### EO LOOP ##########



if __name__ == '__main__':
    main()
