import model
import numpy as np
import os
import model_utils
from tqdm import trange

DEFAULT_WORLD = 0
DEFAULT_EPOCHS = 1
DEFAULT_VERBOSE = "y"
EPOCHS_KEY = "EPOCHS"

def get_user_input(prompt, default_value):
    user_input = input(prompt)
    return user_input.strip() or default_value


def main():

	if not (os.path.exists(f"./api_key/key.json")):
		print("JSON file for credentials does not exist.")
		exit()


	operation_mode = str(input("\noption 't' is train (default)\noption 'c' is train-cycle\noption 'e' is exploit\n\nENTER OPTION: ") or "t")

	if operation_mode == "t":

		world = int(get_user_input("\nwhich World [0-10] would you like to train on? (default is World 0)\nWORLD: ", str(DEFAULT_WORLD)))
		epochs = int(get_user_input(f"\nhow many epochs would you like to train the agent on World {world} for? (default is 1 epoch)\n{EPOCHS_KEY}: ", str(DEFAULT_EPOCHS)))		
		verbose = get_user_input(f"\nverbosity? (default is yes)\n([y]/n)? ", DEFAULT_VERBOSE).lower() == "y"
		verbose_output = True if verbose == "y" else False

		print(f"\ntraining starts from  for {epochs} on world {world}")

		epsilon = 0.9
		q_table = model_utils.init_q_table()

		if not (os.path.exists(f"./results/world_{world}/")):
			os.makedirs(f"./results/world_{world}/")

		current_run_index = len([i for i in os.listdir(f"results/world_{world}")])
		file_path = f"./results/Q-table_world_{world}"

		rewarding_states = []
		penalizing_states = []
		obstacles = []


		for epoch in range(epochs):
			print("EPOCH #"+str(epoch)+":\n\n")
			q_table, rewarding_states, penalizing_states, obstacles = model.learn(
				q_table, worldId=world, mode='train', learning_rate=0.0001, gamma=0.9, epsilon=epsilon, rewarding_states=rewarding_states, penalizing_states=penalizing_states,
				epoch=epoch, obstacles=obstacles, current_run_index=current_run_index, verbose=verbose_output)

			epsilon = model_utils.epsilon_decay(epsilon, epoch, epochs)

			np.save(file_path, q_table)
		np.save(f"./results/obstacles_world_{world}", obstacles)
		np.save(f"./results/rewarding_states_world_{world}", rewarding_states)
		np.save(f"./results/penalizing_states_world_{world}", penalizing_states)

	elif operation_mode == "e":
		
		world = int(get_user_input("\nwhich World [0-10] would you like the agent to exploit? (default is World 0)\nWORLD: ", str(DEFAULT_WORLD)))
		epochs = int(get_user_input(f"\nhow many times would you like the agent to run on World {world} for? (default is 1 time)\nEPOCHS: ", "1"))		
		verbose = get_user_input(f"\nverbosity? (default is yes)\n([y]/n)? ", DEFAULT_VERBOSE).lower() == "y"
		verbose_output = True if verbose == "y" else False

		print(f"\nExploiting world {world} for {epochs} iterations")

		file_path = f"./results/Q-table_world_{world}"
		q_table = np.load(file_path+".npy")

		obstacles = np.load(f"./results/obstacles_world_{world}"+".npy")
		rewarding_states = np.load(f"./results/rewarding_states_world_{world}"+".npy")
		penalizing_states = np.load(f"./results/penalizing_states_world_{world}"+".npy")

		obstacles = obstacles.tolist()
		rewarding_states = rewarding_states.tolist()
		penalizing_states = penalizing_states.tolist()

		epsilon = 0.9
		current_run_index = len([i for i in os.listdir(f"results/world_{world}")])

		for epoch in range(epochs):
			print("EPOCH #"+str(epoch)+":\n\n")
			q_table, rewarding_states, penalizing_states, obstacles = model.learn(
				q_table, worldId=world, mode='expl', learning_rate=0.0001, gamma=0.9, epsilon=epsilon, rewarding_states=rewarding_states, penalizing_states=penalizing_states,
				epoch=epoch, obstacles=obstacles, current_run_index=current_run_index, verbose=verbose_output)
	

	if operation_mode == "c":
		
		confirm = get_user_input(f"\nyou've chosen to train the agent on all Worlds [1-10], this could take a while.. (are you sure?)\nProceed ([y]/n)? ", "y")
		cont = get_user_input(f"\nWould you like to continue training from previous runs? (are you sure?)\nProceed ([y]/n)? ", "y")

		verbose = get_user_input(f"\nverbosity? (default is yes)\n([y]/n)? ", "y").lower() == "y"
		verbose_output = True if verbose == "y" else False
		
		if cont.lower() == "y":
			epochs_computed = int(get_user_input(f"\nHow many epochs were used in previous training runs?\nEPOCHS: ", "0"))
			epochs = int(get_user_input(f"\nhow many more epochs would you the agent to train on each World? (default is 10 epochs)\nEPOCHS: ", "10"))
			init_eps = epsilon = model_utils.epsilon_decay(0.9, 6, epochs_computed+epochs)
			
		else:
			epochs = int(get_user_input(f"\nhow many epochs would you the agent to train on each World? (default is 10 epochs)\nEPOCHS: ", "10"))
			epochs_computed = 0
			init_eps = epsilon = 0.9

		if confirm == "y":
			for i in range(10):
				world = i+1

				print(f"\ntraining from scratch for {epochs} on world {world}! \n(visualizations will be saved to './results/world_{world}/')\n(Q-tables will be saved to './results/Q-table_world_{world}'")

				if not (os.path.exists(f"./results/world_{world}/")):
					os.makedirs(f"./results/world_{world}/")

				current_run_index = len([i for i in os.listdir(f"results/world_{world}")])


				file_path = f"./results/Q-table_world_{world}"

				if cont.lower() == 'y':
					rewarding_states = np.load(open(f"./results/rewarding_states_world_{world}.npy", "rb"))
					penalizing_states = np.load(open(f"./results/penalizing_states_world_{world}.npy", "rb"))
					obstacles = np.load(open(f"./results/obstacles_world_{world}.npy", "rb"))

					q_table = np.load(open(f"./results/Q-table_world_{world}.npy", "rb"))
				else:
					rewarding_states = []
					penalizing_states = []
					obstacles = []
					q_table = model_utils.init_q_table()
				
				t = trange(epochs, desc='Training on all worlds', leave=True)

				for epoch in t:
					t.set_description('Current World={}'.format(i+1))

					print("EPOCH #"+str(epoch)+":\n\n")
					q_table, rewarding_states, penalizing_states, obstacles = model.learn(
						q_table, worldId=world, mode='train', learning_rate=0.0001, gamma=0.9, epsilon=epsilon, rewarding_states=rewarding_states, penalizing_states=penalizing_states,
						epoch=epoch, obstacles=obstacles, current_run_index=current_run_index, verbose=verbose_output)
					
					epsilon = model_utils.epsilon_decay(init_eps, epoch+epochs_computed, epochs+epochs_computed)

					np.save(file_path, q_table)

				np.save(f"./results/obstacles_world_{world}", obstacles)
				np.save(f"./results/rewarding_states_world_{world}", rewarding_states)
				np.save(f"./results/penalizing_states_world_{world}", penalizing_states)

		else:
			exit()

	else:
		print("that option doesn't exist yet :'(")
		exit()

if __name__ == "__main__":
    main()