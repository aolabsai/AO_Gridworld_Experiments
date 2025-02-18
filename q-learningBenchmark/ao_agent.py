import ao_core as ao
import random
import numpy as np
import matplotlib.pyplot as plt


###ARCH



import ao_arch as ar



number_qa_neurons = 15

description = "Q-learningBenchmark"
arch_i = [1, 1, 1, 1, 1, 1]     # 6 neurons corresponding to co-ordinates on the grid
arch_z = [2]           # corresponding to which direction we should move
arch_c = [0]           # adding 1 control neuron which we'll define with the instinct control function below
arch_qa = [number_qa_neurons]

connector_function = "full_conn"


# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, arch_qa=arch_qa, qa_conn="full", description=description)


#Adding Aux Action
def qa0_firing_rule(INPUT, Agent): 
    if Agent.reset_qa:
        print("reset of qa at step: ", steps)
        Agent.counter = number_qa_neurons
        Agent.reset_qa = False
    if not hasattr(Agent, 'counter'):
        Agent.__setattr__("counter", 20)


    # if Agent.counter == 0:
    #     group_response = np.ones(number_qa_neurons)
        

    if Agent.counter < (number_qa_neurons+1):
        Agent.counter -= 1
        group_response = np.zeros(number_qa_neurons)
        group_response[0 : Agent.counter] = 1

            

    else:    #If the agent did not react then dont touch the counter
        group_response = np.zeros(number_qa_neurons)
        group_response[0 : Agent.counter] = 1

     

    group_meta = np.ones(number_qa_neurons, dtype="O")
    group_meta[:] = "qa0"
    return group_response, group_meta
# Saving the function to the Arch so the Agent can access it
Arch.datamatrix_aux[2] = qa0_firing_rule

####END of Arch








# Grid environment setup
grid_size = 5
start = (0, 0)
goal = (4, 4)
num_obs = 3

# Generate obstacles
obs = set()
while len(obs) < num_obs:
    obstacle = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    if obstacle != start and obstacle != goal:
        obs.add(obstacle)

# Action mapping
action_mapping = {
    (0, 0): (-1, 0),   # Move up
    (1, 0): (1, 0),    # Move down
    (0, 1): (0, -1),   # Move left
    (1, 1): (0, 1)     # Move right
}

# Encode position in binary
def encode_position_binary(x, y):
    x_bin = format(x, '03b')
    y_bin = format(y, '03b')
    return [int(bit) for bit in x_bin + y_bin]

# Check if a position is valid
def is_valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size and pos not in obs

# Visualization function
def visualize_grid(path):
    grid = np.zeros((grid_size, grid_size))

    # Mark start, goal, and obstacles
    grid[start] = 0.5  # Start
    grid[goal] = 0.8  # Goal
    for obstacle in obs:
        grid[obstacle] = 1  # Obstacle

    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    # Plot the path
    for (x, y) in path:
        ax.text(y, x, '●', ha='center', va='center', color='red')

    plt.title("Path taken by agent")
    plt.show()



# Initialize agent
agent = ao.Agent(Arch)
agent.reset_qa = False

epidodes = 100
steps_per_episodes = []
plt.ion()
for i in range(epidodes):
    visited_states = [] #keeping track of agent positions

    # Solve the grid
    timed_out = False
    solved = False
    state = start
    steps = 0
    path = [start]  # Track the path
    while not solved and not timed_out:
        steps += 1

        input_to_agent = encode_position_binary(*state)
        response = agent.next_state(input_to_agent, DD=False).tolist()
        response_tuple = tuple(response)



        if response_tuple in action_mapping:
            dx, dy = action_mapping[response_tuple]
            new_state = (state[0] + dx, state[1] + dy)

                        
            if new_state in visited_states:
                print("state already visited")
                state = start
                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    next_position = (state[0] + dx, state[1] + dy)
                    if is_valid(next_position):
                        valid_labels.append(label)
                if valid_labels:
                    label = random.choice(valid_labels)
                else:
                    label = [0, 0]
                agent.next_state(input_to_agent, LABEL=label)  # Send negative feedback
                agent.reset_state()
                path = [start]

            elif not is_valid(new_state):
                # Find a valid action (label) that the agent should take
                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    next_position = (state[0] + dx, state[1] + dy)
                    if is_valid(next_position):
                        valid_labels.append(label)

                if valid_labels:
                    label = random.choice(valid_labels)
                else:
                    label = [0, 0]
                _ = agent.next_state(input_to_agent, label)  # Send feedback
                state = start
                path = [start]  # Reset the path  

            elif steps > 1000:
                solved = True
                print("Failed to solve timed out")
                agent.reset_state()

            elif random.random() < 0.1:  # Random exploration

                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    next_position = (state[0] + dx, state[1] + dy)
                    if is_valid(next_position):
                        valid_labels.append(label)
                if valid_labels:
                    label = random.choice(valid_labels)
                else:
                    label = [0, 0]
            
            elif agent.counter == 0:
                print("pain signal due to qa")
                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    next_position = (state[0] + dx, state[1] + dy)
                    if is_valid(next_position):
                        valid_labels.append(label)
                if valid_labels:
                    label = random.choice(valid_labels)
                else:
                    label = [0, 0]
                agent.reset_qa = True

            elif new_state == goal:
                agent.next_state(input_to_agent, Cpos=True)  # Reward for reaching the goal
                agent.reset_state()
                solved = True
                print("Goal reached in", steps, "steps!")
                path.append(goal)
                agent.reset_qa = True
            else:

                state = new_state
                path.append(state)

            # Loop detection: Track recent positions and check for loops
            visited_states.append(state)
            if len(visited_states) > 6:
                visited_states.pop(0)
            if visited_states.count(state) > 5:
                print("Loop detected! Resetting the agent.")
                state = start
                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    next_position = (state[0] + dx, state[1] + dy)
                    if is_valid(next_position):
                        valid_labels.append(label)
                if valid_labels:
                    label = random.choice(valid_labels)
                else:
                    label = [0, 0]
                agent.next_state(input_to_agent, LABEL=label)  # Send negative feedback
                agent.reset_state()
                print("Loop detected! Resetting the agent.")
                path = [start]  # Reset the path
        else:
            print("Invalid response from the agent.")
            agent.next_state(input_to_agent, Cneg=True)
            agent.reset_state()

    steps_per_episodes.append(steps)
# Visualize the final path

plt.ioff()
plt.plot(steps_per_episodes)
plt.xlabel("Number of trials")
plt.ylabel("Amount of steps")
visualize_grid(path)
