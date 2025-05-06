import ao_core as ao
import random
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

###ARCH


description = "Q-learningBenchmark"
arch_i = [1, 1, 1, 1, 1, 1]     # 6 neurons corresponding to co-ordinates on the grid
arch_z = [2]           # corresponding to which direction we should move
arch_c = [0]           # adding 1 control neuron which we'll define with the instinct control function below

connector_function = "full_conn"


# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ao.Arch(arch_i, arch_z, arch_c, connector_function, description=description)

####END of Arch

def encode_position_binary(x, y):
    x_bin = format(x, '03b')
    y_bin = format(y, '03b')
    return [int(bit) for bit in x_bin + y_bin]

# Visualization function
def visualize_grid(path):
    grid = np.zeros((grid_size, grid_size))

    # Mark start, goal, and obstacles
    grid[start] = 0.5  # Start
    grid[goal] = 0.8  # Goal

    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    # Plot the path
    for (x, y) in path:
        ax.text(y, x, '●', ha='center', va='center', color='red')

    for obstacle in obs:
        ax.text(obstacle[1], obstacle[0], 'X', ha='center', va='center', color='white')

    plt.title("Path taken by agent")
    plt.show()

def map_agent_response_to_direction(response):
    x = response[0]
    y = response[1]
    if (x, y) == (1, 0):
        return (1, 0)   # Move right
    elif (x, y) == (1, 1):
        return (-1, 0)  # Move left
    elif (x, y) == (0, 1):
        return (0, 1)   # Move up
    elif (x, y) == (0, 0):
        return (0, -1)  # Move down
    else:
        print("Invalid response from agent!", response)


def isvalid(pos, prints=True):
    if pos[0] < 0 or pos[0] >= grid_size or pos[1] < 0 or pos[1] >= grid_size:
        if prints:
            print("Invalid move, out of bounds!")
        return False
    if pos in obs:
        if prints:
            print("Invalid move, obstacle detected!")
        return False
    return True

def move_in_random_valid_direction(current_pos):
        possible_moves = [[1, 0], [0, 1], [1, 1], [0,0]]  # need this to retrain the agent

        actucal_moves = [(1,0), (0,1), (-1,0), (0,-1)] # the actual moves we want to make


        valid_moves = []
        for move in actucal_moves:
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if isvalid(next_pos, prints=False):
                valid_moves.append((move))

        #random.shuffle(valid_moves) # Not sure why this is not working

        chosen_move = possible_moves[actucal_moves.index(valid_moves[0])]

        if solved_once:
            agent.next_state(input_to_agent, LABEL=chosen_move)

        return chosen_move

def reset_position(steps):
    plt.ioff()
    number_of_steps_array.append(steps)
    steps = 0
    pos = start
    path = [start]
    agent_inputs = []
    agent_responses = []
    random_exploration = intital_exploration

    return pos, path, agent_inputs, agent_responses, random_exploration, steps


# Grid environment setup
grid_size = 5
start = (0, 0)
goal = (4, 4)
num_obs = 3
obs = []
for i in range(num_obs):
    obsx = (random.randint(0, grid_size-1))
    obsy = (random.randint(0, grid_size-1))

    while (obsx, obsy) == start or (obsx, obsy) == goal or (obsx, obsy) in obs:
        obsx = (random.randint(0, grid_size-1))
        obsy = (random.randint(0, grid_size-1))

    obs.append((obsx, obsy))


solved_once = False # we dont want to give it a pain signal if it has never solved the maze before since it should be randomly exploring


intital_exploration = 0.0 # initial exploration rate- can adjust this as a hyperparameter
episodes = 100
random_exploration = intital_exploration # probability of random exploration
decay = 0.9 # decay factor for the random exploration
number_of_steps_array = []
agent = ao.Agent(Arch)
for i in range(episodes):

    solved = False
    steps = 0
    pos = start
    path = [start]
    

    agent_inputs = []
    agent_responses = [] # keep track of the agent's responses so if it gets better we can give all states a pleasure signal

    while not solved:
        steps += 1

        input_to_agent = encode_position_binary(pos[0], pos[1]) # the input to the agent is the position of the agent in binary

        response = agent.next_state(input_to_agent)
        dx, dy = map_agent_response_to_direction(response)


#        # Move the agent
        candidate_pos = (pos[0] + dx, pos[1] + dy)

        if not isvalid(candidate_pos):
            print(f"Invalid move: {candidate_pos} , moving randomly")
            chosen_move = move_in_random_valid_direction(pos)
            dx, dy = map_agent_response_to_direction(chosen_move)
            candidate_pos = (pos[0] + dx, pos[1] + dy)
            if solved_once:
                agent.next_state(input_to_agent, LABEL=chosen_move)

        if pos == goal:
            solved_once = True # the agent has solved the maze at least once
            solved = True
            if number_of_steps_array:
                if steps < min(number_of_steps_array): # if the agent got better 
                    #agent = ao.Agent(Arch) # maybe we can reset agent so it gets rid of the old
                    for response in agent_responses:
                        agent.next_state(input_to_agent, Cpos=True)

                agent_inputs = []
                agent_responses = []

            print(f"Goal reached in {steps} steps!")
            

        else:
            agent_inputs.append(input_to_agent)
            agent_responses.append(response)
        
        # extra conditions

        ## been to same position before # Cant reallt use this because sometimes the agent needs to go back to the same position to get to the goal, think of a better way to implement
        # if pos in path:
        #     if solved_once:
        #         print("Agent has been to this position before, moving randomly")
        #         chosen_move = move_in_random_valid_direction(pos)

        # if steps > 100:
        #     pos, path, agent_inputs, agent_responses, random_exploration, steps = reset_position(steps)

        if steps > 1000: 
            plt.ioff()
            agent = ao.Agent(Arch)  # Maybe we want reset agent
            #visualize_grid(path) # if the agent is taking too long to solve the maze, we can visualize the path taken so far
            pos, path, agent_inputs, agent_responses, random_exploration, steps = reset_position(steps)



        if random.random() < random_exploration:  # 10% chance to randomly change direction
            possible_moves = [[1, 0], [0, 0], [0, 1], [1, 1]]

            candidate_pos = move_in_random_valid_direction(pos)



        pos = candidate_pos
        path.append(pos)




    number_of_steps_array.append(steps)
    random_exploration *= decay  # Decay the exploration rate


print("Final path taken by the agent:", path)
plt.ioff()
visualize_grid(path)


plt.figure(figsize=(10, 5))
plt.title("Number of steps taken to reach the goal")
plt.xlabel("Episode")
plt.plot(number_of_steps_array)

plt.show()