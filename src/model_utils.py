import matplotlib.pyplot as plt
import os
import numpy as np

def init_q_table():
    '''
    init q table (initilizations are all [0])
    defines grid as 40x40, 4 possible actions (N, S, E, W)
    access grid as row, col, action
    ex of indexing: q-tab[0][0][0] gird 0:0, action 'N'
    '''

    return (np.zeros((40, 40, 4)))

def number_to_direction(num):
    """
    Translates the index returned from np.argmax()

    Returns => str: The corresponding movement command ('N', 'S', 'E', 'W') or 'ERROR!' if the index is invalid.
    
    """
    moves = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}
    return moves.get(num, 'ERROR!')


def update_q_table(location, q_table, reward, gamma, new_location, learning_rate, move_index):
    '''
    Function to update q table uses Bellman equation.
    '''

    #collecting the current understanding of the best q value based upon our new location, weight it by gamma and add reward
    right_side = reward + gamma * q_table[new_location[0], new_location[1], :].max() - q_table[location[0], location[1], move_index]

    #use the previous location to 
    updated_q = q_table[location[0], location[1], move_index] + learning_rate * right_side

    #update q_table with new value
    q_table[location[0], location[1], move_index] = updated_q


def plot_learning(worldId, epoch, cumulative_average, rn):
    plt.figure(2)
    plt.plot(cumulative_average)
    plt.xscale('log')
    if not os.path.exists(f'results/world_{worldId}/attempt_{rn}'):
        os.makedirs(f'results/world_{worldId}/attempt_{rn}')
    plt.savefig(f'results/world_{worldId}/attempt_{rn}/world_{worldId}_epoch{epoch}learning.png')

def epsilon_decay(epsilon, epoch, epochs):
    '''
    function to exponentially decrease the episilon value 
    acroccs the total number of epochs we train on
    this leads us to explore less as we progress through epochs 
    '''
    
    epsilon = epsilon*np.exp(-.01*epoch)
    
    print(f"\nNEW EPSILON: {epsilon}\n")
    return epsilon

