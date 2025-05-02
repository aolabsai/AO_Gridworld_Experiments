import ao_core as ao
import random
import matplotlib.pyplot as plt
import numpy as np 

from arch__gridworld import Arch
from config import ao_api_key

def is_valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size and pos not in obs

def update_visualization(positions, current_pos, ax, grid_size, obs, start, goal):
    ax.clear()
    # Set grid ticks and gridlines
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # For each cell, display its binary value 
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] not in obs:
                distance = abs(i - goal[0]) + abs(j - goal[1])
                binary_val = format(distance, '04b')  # 4-bit binary representation ( input to agent!!)
                ax.text(j + 0.5, i + 0.5, binary_val,
                        ha='center', va='center', fontsize=8, color='black')

    # Draw obstacles, start, and goal markers
    for obstacle in obs:
        ax.add_patch(plt.Rectangle((obstacle[1], obstacle[0]), 1, 1, color='black'))
    ax.add_patch(plt.Rectangle((start[1], start[0]), 1, 1, color='green', alpha=0.6, label="Start"))
    ax.add_patch(plt.Rectangle((goal[1], goal[0]), 1, 1, color='blue', alpha=0.6, label="Goal"))

    # Draw the path taken so far (red dots)
    for (x, y) in positions:
        ax.text(y + 0.5, x + 0.5, '•', ha='center', va='center', fontsize=12, color='red')

    # Highlight the current agent position (orange circle)
    ax.add_patch(plt.Circle((current_pos[1] + 0.5, current_pos[0] + 0.5), 0.3, color='orange', label="Agent"))

    ax.set_title("Path Taken by Agent")
    plt.draw()
    plt.pause(0.01)  # Adjust pause duration as needed


def get_binary_dist(pos, goal):
    distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    return [bit for bit in format(distance, "04b")]

# Encode position in binary
def encode_position_binary(x, y):
    x_bin = format(x, '03b')
    y_bin = format(y, '03b')
    return [int(bit) for bit in x_bin + y_bin]

# Grid and environment setup
grid_size = 5
start = [0, 0]
goal = [4, 4]

num_obs = 3
obs = []

while len(obs) < num_obs:
    obstacle = [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]
    if obstacle != start and obstacle != goal and obstacle not in obs:
        obs.append(obstacle)

action_mapping = {
    (0, 0): (-1, 0),   # Move up
    (1, 0): (1, 0),    # Move down
    (0, 1): (0, -1),   # Move left
    (1, 1): (0, 1)     # Move right
}

# Initialize the agent
agent = ao.Agent(Arch)
episodes = 20
step_array = []




# Set up Matplotlib for interactive updating
plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))

for episode in range(episodes):
    pos = start
    last_pos = start
    positions = [pos.copy()]
    not_solved = True
    steps = 0

    # Initial visualization
    update_visualization(positions, pos, ax, grid_size, obs, start, goal)
    
    while not_solved:
        # Get the status of surrounding cells
        steps +=1
        surroundings = [
            [pos[0]-1, pos[1]],   # up
            [pos[0], pos[1]-1],   # left
            [pos[0]+1, pos[1]],   # down
            [pos[0], pos[1]+1]    # right
        ]

        input_signal = []
        for i, v in enumerate(surroundings):
            if v in obs:
                input_signal.append(1)
            else:
                input_signal.append(0)

        x, y = pos

        input_signal += get_binary_dist(pos, goal)  # Add the gradient point - practically coords i guess

        # Agent decides the next move
        response = agent.next_state(input_signal)
        response_tuple = tuple(response)

        if response_tuple in action_mapping:
            dx, dy = action_mapping[response_tuple]
            new_pos = [pos[0] + dx, pos[1] + dy]
        else:
            new_pos = pos.copy()


        if pos == goal:
            step_array.append(steps)
            print("Reached goal!")
            positions.append(pos.copy())
            #update_visualization(positions, pos, ax, grid_size, obs, start, goal)
            not_solved = False
            agent.reset_state()


        # Check if the new position is valid

        elif not is_valid(new_pos):
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                candidate = [pos[0] + dx, pos[1] + dy]
                if candidate != pos:
                    valid_labels.append(label)
            chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
            agent.next_state(input_signal, chosen_label)  # Send feedback
            new_pos = pos.copy()  # Revert to last valid position

        elif is_valid(new_pos):
            pos = new_pos.copy()


        elif pos == last_pos:
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                candidate = [pos[0] + dx, pos[1] + dy]
                if candidate != pos:
                    valid_labels.append(label)
            chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
            agent.next_state(input_signal, LABEL=chosen_label)
            agent.reset_state()


        num = 0
        for i in range(len(positions)):
            if positions[i] == pos:
                num += 1
        if num > 5:
            print("loop")
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                candidate = [pos[0] + dx, pos[1] + dy]
                if candidate != pos:
                    valid_labels.append(label)
            chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
            agent.next_state(input_signal, LABEL=chosen_label)
            agent.reset_state()

                    
        positions.append(pos.copy())
        #update_visualization(positions, pos, ax, grid_size, obs, start, goal)
        last_pos = pos.copy()

print("complted")
plt.ioff()
plt.show()
print("step array: ", step_array)

plt.plot(step_array)
plt.show()
