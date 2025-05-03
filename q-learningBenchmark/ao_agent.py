import ao_core as ao
import random
import numpy as np
import matplotlib.pyplot as plt
import time 
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


def isvalid(pos):
    if pos[0] < 0 or pos[0] >= grid_size or pos[1] < 0 or pos[1] >= grid_size:
        print("Invalid move, out of bounds!")
        return False
    if pos in obs:
        print("Invalid move, obstacle detected!")
        return False

    return True

def move_in_random_valid_direction(current_pos):
        possible_moves = [[1, 0], [0, 1], [1, 1], [0,0]]  # imagne the agent output here. This would need to use map_agent_response_to_direction to get the actual move

        valid_moves = []
        for move in possible_moves:
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if isvalid(next_pos):
                valid_moves.append((move))


        
        chosen_move = valid_moves[0]

        if solved_once:
            agent.next_state(input_to_agent, LABEL=chosen_move)

        print(f"Moving randomly to {chosen_move}")
        return chosen_move



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


solved_once = False # we dont want to give it a pain signal if it has never solved the maze before

episodes = 10
number_of_steps_array = []
for i in range(episodes):

    solved = False
    steps = 0
    pos = start
    path = [start]
    agent = ao.Agent(Arch)

    agent_inputs = []
    agent_responses = [] # keep track of the agent's responses so if it gets better we can give all states a pleasure signal

    while not solved:
        steps += 1

        input_to_agent = encode_position_binary(pos[0], pos[1])

        response = agent.next_state(input_to_agent)
        dx, dy = map_agent_response_to_direction(response)


#        # Move the agent
        candidate_pos = (pos[0] + dx, pos[1] + dy)

        if not isvalid(candidate_pos):
            print("Invalid move, moving randomly")
            print("atteempting to move to", candidate_pos)
            chosen_move = move_in_random_valid_direction(pos)
            dx, dy = map_agent_response_to_direction(chosen_move)
            candidate_pos = (pos[0] + dx, pos[1] + dy)
            if solved_once:
                agent.next_state(input_to_agent, LABEL=chosen_move)

        pos = candidate_pos

        if pos == goal:
            solved_once = True # the agent has solved the maze at least once
            solved = True
            if number_of_steps_array:
                if steps < min(number_of_steps_array): # if the agent got better 
                    agent = ao.Agent(Arch) # reset the agent
                    print("Agent got better, restting with new better agent")
                    for response in agent_responses:
                        agent.next_state(input_to_agent, Cpos=True)

            print(f"Goal reached in {steps} steps!")
            

        else:
            agent_inputs.append(input_to_agent)
            agent_responses.append(response)
        
        # extra conditions

        ## been to same position before
        if pos in path:
            if solved_once:
                print("Agent has been to this position before, moving randomly")
                chosen_move = move_in_random_valid_direction(pos)

        if steps > 100:
            plt.ioff()
            visualize_grid(path)
            agent = ao.Agent(Arch)
            print("Agent has taken too long, resetting")
            steps = 0
            pos = start
            path = [start]
            agent_inputs = []
            agent_responses = []



        if random.random() < 0.05:  # 10% chance to randomly change direction
            possible_moves = [[1, 0], [0, 0], [0, 1], [1, 1]]

            valid_moves = []
            for move in possible_moves:
                next_pos = (pos[0] + move[0], pos[1] + move[1])
                if isvalid(next_pos):
                    valid_moves.append((move, next_pos))

            chosen_move, candidate_pos = random.choice(valid_moves)

            if solved_once:
                agent.next_state(input_to_agent, LABEL=chosen_move)


        path.append(pos)
    number_of_steps_array.append(steps)



print("Final path taken by the agent:", path)
plt.ioff()
visualize_grid(path)


plt.figure(figsize=(10, 5))
plt.title("Number of steps taken to reach the goal")
plt.xlabel("Episode")
plt.plot(number_of_steps_array)

plt.show()