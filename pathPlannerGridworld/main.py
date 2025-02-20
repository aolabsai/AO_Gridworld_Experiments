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

    for obstacle in obs:
        ax.add_patch(plt.Rectangle((obstacle[1], obstacle[0]), 1, 1, color='black'))

    ax.add_patch(plt.Rectangle((start[1], start[0]), 1, 1, color='green', alpha=0.6, label="Start"))
    ax.add_patch(plt.Rectangle((goal[1], goal[0]), 1, 1, color='blue', alpha=0.6, label="Goal"))

    # Draw the path taken so far (red dots)
    for (x, y) in positions:
        ax.text(y+0.5, x+0.5, '•', ha='center', va='center', fontsize=12, color='red')

    # Highlight the current agent position (orange circle)
    ax.add_patch(plt.Circle((current_pos[1]+0.5, current_pos[0]+0.5), 0.3, color='orange', label="Agent"))

    ax.set_title("Path Taken by Agent")
    plt.draw()
    plt.pause(0.3)  # Adjust pause duration as needed

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
episodes = 5

# Set up Matplotlib for interactive updating
plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))

for episode in range(episodes):
    pos = start
    last_pos = start
    positions = [pos.copy()]
    not_solved = True

    # Initial visualization
    update_visualization(positions, pos, ax, grid_size, obs, start, goal)
    
    while not_solved:
        # Get the status of surrounding cells
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

        # Agent decides the next move
        response = agent.next_state(input_signal)
        response_tuple = tuple(response)

        if response_tuple in action_mapping:
            dx, dy = action_mapping[response_tuple]
            new_pos = [pos[0] + dx, pos[1] + dy]
        else:
            new_pos = pos.copy()

        # Check if the new position is valid
        if not is_valid(new_pos):
            print("Invalid move attempted:", new_pos)
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                candidate = [pos[0] + dx, pos[1] + dy]
                if is_valid(candidate):
                    valid_labels.append(label)
            chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
            agent.next_state(input_signal, chosen_label)  # Send feedback
            new_pos = pos.copy()  # Revert to last valid position
        else:
            pos = new_pos

        # Check for goal and other feedback conditions
        if pos == goal:
            print("Reached goal!")
            positions.append(pos.copy())
            update_visualization(positions, pos, ax, grid_size, obs, start, goal)
            not_solved = False
            agent.reset_state()


        elif pos == last_pos:
            print("No movement")
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                candidate = [pos[0] + dx, pos[1] + dy]
                if is_valid(candidate):
                    valid_labels.append(label)
            chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
            agent.next_state(input_signal, LABEL=chosen_label)
            agent.reset_state()


            
        elif pos in positions:
            num = 0
            print("positions: ", positions)
            for i in range(len(positions)):
                if positions[i] == pos:
                    num += 1
            if num > 5:
                print("loop")
                valid_labels = []
                for label, (dx, dy) in action_mapping.items():
                    candidate = [pos[0] + dx, pos[1] + dy]
                    if is_valid(candidate):
                        valid_labels.append(label)
                chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
                agent.next_state(input_signal, LABEL=chosen_label)
                agent.reset_state()


        elif (abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])) > (abs(goal[0] - last_pos[0]) + abs(goal[1] - last_pos[1])):
            valid_labels = []
            for label, (dx, dy) in action_mapping.items():
                candidate = [pos[0] + dx, pos[1] + dy]
                if is_valid(candidate):
                    valid_labels.append(label)
            chosen_label = random.choice(valid_labels) if valid_labels else (0, 0)
            agent.next_state(input_signal, LABEL=chosen_label)  # Send negative feedback
            agent.reset_state()
            print("got futher")

            
        elif (abs(goal[0] - last_pos[0]) + abs(goal[1] - last_pos[1])) > (abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])):
            print("Got closer")
            agent.next_state(input_signal, Cpos=True)
                    

        print("Current position:", pos)
        positions.append(pos.copy())
        update_visualization(positions, pos, ax, grid_size, obs, start, goal)
        last_pos = pos.copy()

plt.ioff()
plt.show()
