import ao_core as ao
import random
import matplotlib.pyplot as plt

import numpy as np 

from arch__gridworld import Arch



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
    for [x, y] in path:
        ax.text(y, x, '●', ha='center', va='center', color='red')

    plt.title("Path taken by agent")
    plt.show()


def is_valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size and pos not in obs

grid_size = 5

start = [0,0]
goal=[4,4]
num_obs = 3
not_solved = True

obs = []
while len(obs) < num_obs:
    obstacle = [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]
    if obstacle != start and obstacle != goal:
        obs.append(obstacle)

action_mapping = {
    (0, 0): (-1, 0),   # Move up
    (1, 0): (1, 0),    # Move down
    (0, 1): (0, -1),   # Move left
    (1, 1): (0, 1)     # Move right
}



agent = ao.Agent(Arch)
episodes = 5

for i in range(episodes):
    pos = start
    steps = 0
    last_pos = start
    not_solved = True
    positions = []

    while not_solved:
        steps = steps +1

        

        surrondings = [[pos[0]-1, pos[1]], [pos[0], pos[1]-1], [pos[0]+1, pos[1]], [pos[0], pos[1]+1]]

        input = []
        for i, v in enumerate(surrondings):
            if v in obs:
                input.append(1)
            else:
                input.append(0)


        response = agent.next_state(input)

        response_tuple = tuple(response)



        if response_tuple in action_mapping:
            x, y = action_mapping[response_tuple]

            pos = [x + pos[0], y + pos[1]] 



        if not is_valid(pos):
                # Find a valid action (label) that the agent should take
                print("not valid: ", pos)
                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    next_position = (pos[0] + dx, pos[1] + dy)
                    if is_valid(next_position):
                        valid_labels.append(label)

                if valid_labels:
                    label = random.choice(valid_labels)
                else:
                    label = [0, 0]
                agent.next_state(input, label)  # Send feedback
                pos= last_pos

        elif pos == goal:
            print("solved")
            not_solved = False

            visualize_grid(positions)



        elif (abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])) > (abs(goal[0] - last_pos[0]) + abs(goal[1] - last_pos[1])):
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                next_position = (pos[0] + dx, pos[1] + dy)
                if is_valid(next_position):
                    valid_labels.append(label)
            if valid_labels:
                label = random.choice(valid_labels)
            else:
                label = [0, 0]
            agent.next_state(input, LABEL=label)  # Send negative feedback
            agent.reset_state()

        elif (abs(goal[0] - last_pos[0]) + abs(goal[1] - last_pos[1]) >(abs(goal[0] - pos[0]) + abs(goal[1] - pos[1]))):
            print("got closer")
            agent.next_state(input, Cpos = True)


        print(pos)
        positions.append(pos)
        last_pos = pos




        