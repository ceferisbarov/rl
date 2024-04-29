import numpy as np
import api
import display
from matplotlib import pyplot

import model_utils



def learn(q_table, worldId=0, mode='train', learning_rate=0.001, gamma=0.9, epsilon=0.9, rewarding_states=[], penalizing_states=[], epoch=0, obstacles=[], current_run_index=0, verbose=True):
    '''
    Learning function
    returns: q_table [NumPy Array], rewarding_states [list], penalizing_states [list], obstacles [list]
    '''
    #create the api instance
    api_instance = api.API(worldId=worldId)
    world_response = api_instance.enter_world()

    if verbose: print("world_response: ",world_response)

    #init terminal state reached
    terminal_state = False

    #create a var to track the type of terminal state
    is_positive_terminal = False

    #accumulate the rewards so far for plotting reward over step
    rewards_acquired = []

    #find out where we are
    location_response = api_instance.locate_agent()

    #create a list of everywhere we've been for the viz
    visited = []

    if verbose: print("location_response",location_response)
    
    #OK response looks like {"code":"OK","world":"0","state":"0:2"}
    if location_response["code"] != "OK":
            print(f"something broke on locate_agent call \nresponse lookes like: {location_response}")
            return -1
    
    # convert JSON into a tuple (x,y)
    location = int(location_response["state"].split(':')[0]), int(location_response["state"].split(':')[1]) #location is a tuple (x, y)
    
    # SET UP FIGURE FOR VISUALIZATION.
    pyplot.figure(1, figsize=(10,10))
    current_board = [[float('-inf')] * 40 for temp in range(40)]
    
    #keep track of where we've been for the visualization
    visited.append(location)
    while True:
        #////////////////// CODE FOR VISUALIZATION
        current_board[location[1]][location[0]] = 1
        for i in range (len(current_board)):
            for j in range(len(current_board)):
                if (current_board[i][j] != 0):
                    current_board[i][j] -= .1
        for obstacle in obstacles:
            if obstacle in visited:
                obstacles.remove(obstacle)
        print(current_board)
        display.update_grid(current_board, rewarding_states, penalizing_states, obstacles, current_run_index, epoch, worldId, location, verbose)
        #//////////////// END CODE FOR VISUALIZATION

        #in q-table, get index of best option for movement based on our current state in the world
        if mode == 'train':
            #use an episolon greedy approach to randomly explore or exploit
            if np.random.uniform() < epsilon:
                unexplored = np.where(q_table[location[0]][location[1]].astype(int) == 0)[0]
                explored = np.where(q_table[location[0]][location[1]].astype(int) != 0)[0]

                if unexplored.size != 0:
                    move_index = int( np.random.choice( unexplored ) )
                else:
                    move_index = int( np.random.choice( explored ) )
            else:
                move_index = np.argmax(q_table[location[0]][location[1]])

        else:
            #mode is exploit -we'll use what we already have in the q-table to decide on our moves
            move_index = np.argmax(q_table[location[0]][location[1]])

        #make the move - transition into a new state
        move_response = api_instance.make_move(move=model_utils.number_to_direction(move_index), worldId=str(worldId)) 

        if verbose: print("move_response", move_response)
        
        if move_response["code"] != "OK":
            #handel the unexpected
            print(f"something broke on make_move call \nresponse lookes like: {move_response}")

            move_failed = True
            while move_failed:
                move_response = api_instance.make_move(move=model_utils.number_to_direction(move_index), worldId=str(worldId))

                print("\n\ntrying move again!!\n\n")

                if move_response["code"] == 'OK':
                    move_failed = False
        
        # check that we're not in a terminal state, and if not convert new location JSON into tuple
        if move_response["newState"] is not None:
            #we're now in new_location, which will be a tuple of where we are according to the API
            #KEEP IN MIND the movment of our agent is apparently STOCHASTIC
            new_location = int(move_response["newState"]["x"]), int(move_response["newState"]["y"]) #tuple (x,y)
            
            # keep track of if we hit any obstacles
            expected_loc = list(location)

            #convert the move we tried to make into an expected location where we think we'll end up (expected_loc) 
            recent_move = model_utils.number_to_direction(move_index)
      
            if recent_move == "N":
                expected_loc[1]+=1
            elif recent_move == "S":
                expected_loc[1]-=1
            elif recent_move == "E":
                expected_loc[0]+=1
            elif recent_move == "W":
                expected_loc[0]-=1


            expected_loc = tuple(expected_loc)

            if verbose: print(f"New Loc: {new_location} (where we actually are now):")
            if verbose: print(f"Expected Loc: {expected_loc} (where we thought we were going to be):")

            if (mode == "train"):
                obstacles.append(expected_loc)

            #continue to track where we have been
            visited.append(new_location)

            #if we placed an obstacle there in the vis, remove it
            for obstacle in obstacles:
                if obstacle in visited:
                    obstacles.remove(obstacle)
            
            
        else:
            #we hit a terminal state
            terminal_state = True
            print("\n\n--------------------------\nTERMINAL STATE ENCOUNTERED\n--------------------------\n\n")
       
        #get the reward for the most recent move we made
        reward = float(move_response["reward"])


        #add reward to plot
        rewards_acquired.append(reward) 

        #if we are training the model then update the q-table for the state we were in before
        #using the bellman-human algorithim
        if mode == "train":
            model_utils.update_q_table(location, q_table, reward, gamma, new_location, learning_rate, move_index)
        
        #update our current location variable to our now current location
        location = new_location


        #if we are in a terminal state then we need to collect the information for our visualization
        #and we need to end our current training epoch
        if terminal_state:
            print(f"Terminal State REWARD: {reward}")

            if reward > 0:
                #we hit a positive reward so keep track of it as a good reward terminal-state
                is_positive_terminal = True
            if not(location in rewarding_states) and not(location in penalizing_states):
                #update our accounting of good and bad terminal states for the visualization
                if is_positive_terminal:
                    rewarding_states.append(location)
                else:
                    penalizing_states.append(location)

            print(current_board)
            #update our visualization a last time before moving onto the next epoch
            display.update_grid(current_board, rewarding_states, penalizing_states, obstacles, current_run_index, epoch, worldId, location, verbose)
            break

    #possibly not needed but this seperates out the plot
    pyplot.figure(2, figsize=(5,5))
    #cumulative average for plotting reward by step over time purposes
    cumulative_average = np.cumsum(rewards_acquired) / (np.arange(len(rewards_acquired)) + 1)
    # plot reward over each step of the agent
    model_utils.plot_learning(worldId, epoch, cumulative_average, current_run_index)

    return q_table, rewarding_states, penalizing_states, obstacles

