import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def updateAnim(frame, ax, positions, Map):
    Gmin, Gmax, Gblocked = Map
    # Clear the previous frame
    ax.clear()

    # Draw the grid
    ax.set_xticks(range(Gmin-1, Gmax+1))
    ax.set_yticks(range(Gmin-1, Gmax+1))
    ax.set_xticks(np.linspace(Gmin-1-0.5, Gmax+1-0.5, Gmax-Gmin+3), minor=True)
    ax.set_yticks(np.linspace(Gmin-1-0.5, Gmax+1-0.5, Gmax-Gmin+3), minor=True)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5)

    # Draw the gray shade for blocked positions
    for x, y in Gblocked:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='gray'))

    # Draw the red circle
    ax.plot(positions[frame][0], positions[frame][1], 'ro', markersize=15)

    # Draw the blue triangle
    ax.plot(positions[frame][2], positions[frame][3], 'b^', markersize=15)

    # Set axis limits
    ax.set_xlim(Gmin-0.5, Gmax+0.5)
    ax.set_ylim(Gmin-0.5, Gmax+0.5)

def createAnimation(positions, Map):
    Gmin, Gmax, Gblocked = Map
    # Create a figure
    fig, ax = plt.subplots()

    # Create the animation
    ani = animation.FuncAnimation(fig, updateAnim, fargs=(ax, positions, Map), frames=len(positions), interval=1000, repeat=False)

    # Display the animation
    plt.show()
