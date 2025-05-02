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

    plt.title("Path taken by agent")
    plt.show()


# Grid environment setup
grid_size = 5
start = (0, 0)
goal = (4, 4)

episodes = 100
number_of_steps_array = []
for i in range(episodes):

    solved = False
    steps = 0
    pos = start
    path = [start]
    agent = ao.Agent(Arch)


    while not solved:
        steps += 1

        input_to_agent = encode_position_binary(pos[0], pos[1])

        response = agent.next_state(input_to_agent)

        x_response = response[0]
        y_response = response[1]

        new_pos = (pos[0] + x_response, pos[1] + y_response)


        # Ensure the agent stays within the grid boundaries
        if new_pos[0] < 0 or new_pos[0] >= grid_size or new_pos[1] < 0 or new_pos[1] >= grid_size:
            print("Out of bounds!")
        else:
            pos = new_pos

        if pos == goal:
            solved = True
            agent.next_state(input_to_agent, Cpos=True)
            print("Goal reached in", steps, "steps!")
            number_of_steps_array.append(steps)

        else:
            print(f"Step {steps}: Moved to {new_pos}")

        path.append(pos)



print("Final path taken by the agent:", path)
plt.ioff()
visualize_grid(path)


plt.figure(figsize=(10, 5))
plt.title("Number of steps taken to reach the goal")
plt.xlabel("Episode")
plt.plot(number_of_steps_array)

plt.show()