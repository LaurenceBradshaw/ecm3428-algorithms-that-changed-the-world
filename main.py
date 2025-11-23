import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from matplotlib import colors
import time
import os
from typing import Iterator, Callable

Coord = tuple[int, int]
Frame = tuple[Coord, set[Coord], dict[Coord, int]]

DIRECTIONS = [(-1,-1),(0,-1),(1,-1),
              (-1, 0),       (1, 0),
              (-1, 1),(0, 1),(1, 1)]

##################################################
# Utilities
##################################################

def heuristic(a: Coord, b: Coord) -> float:
    """
    Euclidean distance heuristic

    Args:
        a: coordinates
        b: coordinates

    Returns:
        Euclidean distance between the two points
    """
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def path_length(path: list[Coord]) -> float:
    """
    Calculate the path length

    Args:
        path: list of coordinates that make up the path

    Returns:
        The path length
    """
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        length += heuristic(path[i], path[i + 1])
    return length

def load_map(path: str) -> np.ndarray:
    """
    Load the map file

    Args:
        path: file path of the map to load

    Returns:
        Loaded map in a numpy array
    """
    height = None
    width = None

    with open(path, "r") as file_handle:
        # read metadata and then read the map
        for line in file_handle:
            stripped = line.strip().lower()

            if stripped.startswith("height"):
                parts = stripped.split()
                height = int(parts[1])

            elif stripped.startswith("width"):
                parts = stripped.split()
                width = int(parts[1])

            elif stripped == "map":
                break

        if height is None or width is None:
            raise ValueError("Missing height or width in map header")

        grid = np.zeros((height, width), dtype=np.int8)

        row_index = 0
        for line in file_handle:
            line = line.rstrip("\n")
            if not line:
                continue

            if len(line) != width:
                raise ValueError("Map line length does not match declared width")

            col_index = 0
            for ch in line:
                if ch == ".":
                    grid[row_index, col_index] = 0  # free space
                elif ch == "S":
                    grid[row_index, col_index] = 1  # shallow water
                elif ch == "T":
                    grid[row_index, col_index] = 2  # tree
                elif ch == "W":
                    grid[row_index, col_index] = 3  # water
                elif ch == "@":
                    grid[row_index, col_index] = 4  # out of bounds
                else:
                    raise ValueError(f"Unknown map symbol: {ch}")
                col_index += 1

            row_index += 1

        if row_index != height:
            raise ValueError("Number of map rows does not match declared height")

    return grid

def trim_grid(grid: np.ndarray, start: Coord, goal: Coord, pad: int=0) -> tuple[np.ndarray, Coord, Coord]:
    """
    Trim the grid down to reduce the search space. It is massive otherwise

    Args:
        grid: the grid to trim
        start: the start coordinate
        end: the end coordinate
        pad: padding to ensure enough of the map is retained for a feasable path

    Returns:
        tuple containing the trimmed grid and remapped start and goal coords
    """
    start_c, start_r = start
    goal_c, goal_r = goal

    # initial bounding box
    min_r = min(start_r, goal_r) - pad
    max_r = max(start_r, goal_r) + pad
    min_c = min(start_c, goal_c) - pad
    max_c = max(start_c, goal_c) + pad

    # clamp to grid bounds
    min_r = max(min_r, 0)
    min_c = max(min_c, 0)
    max_r = min(max_r, grid.shape[0] - 1)
    max_c = min(max_c, grid.shape[1] - 1)

    # compute current height and width
    height = max_r - min_r + 1
    width = max_c - min_c + 1

    # make it square by expanding the shorter side
    if height > width:
        diff = height - width
        min_c = max(min_c - diff // 2, 0)
        max_c = min(max_c + (diff - diff // 2), grid.shape[1] - 1)
    elif width > height:
        diff = width - height
        min_r = max(min_r - diff // 2, 0)
        max_r = min(max_r + (diff - diff // 2), grid.shape[0] - 1)

    # slice the grid
    trimmed = grid[min_r:max_r + 1, min_c:max_c + 1]

    # remap coordinates
    new_start = (start_c - min_c, start_r - min_r)
    new_goal = (goal_c - min_c, goal_r - min_r)

    return trimmed, new_start, new_goal

def traversable(grid: np.ndarray, x: int, y: int) -> bool:
    """
    Checks if it is possible to travel to the given x,y coords.
    Handles index out of bounds as well

    Args:
        grid: the grid we are traversing
        x: x coord
        y: y coord

    Returns:
        boolean to indicate if the x,y coords can be traversed
    """
    out_of_bounds = not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0])
    if out_of_bounds:
        return False

    val = grid[y, x]
    if val == 2 or val == 3 or val == 4:
        return False
    else:
        return True
    
def mark_point(display: np.ndarray, point: Coord, value: int, size: int=3) -> None:
    """
    Marks a point with an area. Useful for when the start and goal points appear quite small
    on the animation.

    Args:
        display: the grid
        point: the point to mark
        value: the of the mark
        size: the size of the area
    """
    c, r = point
    rows, cols = display.shape
    half = size // 2
    r_min = max(r - half, 0)
    r_max = min(r + half + 1, rows)
    c_min = max(c - half, 0)
    c_max = min(c + half + 1, cols)
    display[r_min:r_max, c_min:c_max] = value
    
def build_final_display(grid: np.ndarray, start: Coord, goal: Coord, frames: list[Frame]) -> np.ndarray:
    """
    Set the grid up with the final frame.

    Args:
        grid: the grid
        start: start coord
        end: end coord
        frames: animation frames

    Returns:
        the grid with cells set to the final frame state
    """
    _, explored_set, _ = frames[-1]

    display = grid.copy()

    for x, y in explored_set:
        if display[y, x] == 0:
            display[y, x] = 8

    mark_point(display, start, 5, size=3)
    mark_point(display, goal, 6, size=3)

    return display

def save_final(display: np.ndarray, path: list[Coord], title: str, filename: str) -> None:
    """
    Save a plot

    Args:
        display: the grid
        path: the discovered path
        title: figure title
        filename: plot file name
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(display, cmap=cmap, norm=norm, origin='upper')
    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        ax.plot([x1, x2],[y1, y2], color="orange", linewidth=1, alpha=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

##################################################
# Common framework for both A* and JPS
##################################################

def generic_search(grid: np.ndarray, start: Coord, goal: Coord, 
                   neighbour_function: Callable[[np.ndarray, Coord, Coord], Iterator[tuple[Coord, float, list[Coord]]]]
                   ) -> tuple[list[Coord], list[Frame]]:
    """
    A* search with a generic neighbour function so it can run both A* and JPS

    Args:
        grid: the grid to find the path in
        start: the starting point
        goal: the goal point
        neighbour_function: function to generate neighbours

    Returns:
        a tuple containing the path found and animation frames
    """
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    explored = set()
    frames = []
    closed = set()

    while open_list:
        f, g, current, parent = heapq.heappop(open_list)
        if current in came_from:
            continue
        
        closed.add(current)
        came_from[current] = parent
        explored.add(current)
        frames.append((current, explored.copy(), g_score.copy()))  # Animation purposes only

        # Reconstruct path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1], frames

        # Expand neighbours
        for neighbour, cost, intermediate_nodes in neighbour_function(grid, current, goal):
            if neighbour in closed:
                continue
            tentative_g = g + cost
            if neighbour not in g_score or tentative_g < g_score[neighbour]:
                g_score[neighbour] = tentative_g
                f_score = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_list, (f_score, tentative_g, neighbour, current))
                closed.update(intermediate_nodes)

    return [], frames

##################################################
# Neighbour expansion for each variant
##################################################

def a_star_neighbours(grid: np.ndarray, current: Coord, goal: Coord) -> Iterator[tuple[Coord, float, list[Coord]]]:
    """
    Get the neighbours using the A* method. All traversable neighbouring points

    Args:
        grid: the grid we are exploring
        current: the current point
        goal: the goal point (unused)

    Returns:
        the neighbouring points to explore along with their heuristic
    """
    for dx, dy in DIRECTIONS:
        nx, ny = current[0] + dx, current[1] + dy
        if not traversable(grid, nx, ny):
            continue
        
        # Check corner cutting
        if not traversable(grid, nx - dx, ny) or not traversable(grid, nx, ny - dy):
            continue

        yield (nx, ny), heuristic(current, (nx,ny)), []

def jump(grid: np.ndarray, current: Coord, dx: int, dy: int, goal: Coord) -> tuple[Coord, list[Coord]] | tuple[None, list[Coord]]:
    """
    Find jump points in the given direction

    Args:
        grid: the grid we are exploring
        current: the current point
        dx: x direction
        dy: y direction
        goal: the goal point

    Returns:
        Jump point coordinates and intermetdiate nodes explored
    """
    x, y = current
    intermediate_nodes = [] # Hold intermediate nodes so we do not explore the same nodes more than once (will be added to closed set)

    while True:
        x += dx
        y += dy

        if not traversable(grid, x, y):
            return None, intermediate_nodes
        
        # Goal reached
        if (x, y) == goal:
            return (x, y), intermediate_nodes
        
        intermediate_nodes.append((x, y))

        if abs(dx) != abs(dy) and not traversable(grid, x + dx, y + dy):
            return None, intermediate_nodes
        
        # Horizontal
        if dx != 0 and dy == 0:
            behind_below_diag_blocked = not traversable(grid, x - dx, y + 1)
            below_free = traversable(grid, x, y + 1)
            behind_above_diag_blocked = not traversable(grid, x - dx, y - 1)
            above_free = traversable(grid, x, y - 1)
            if (behind_above_diag_blocked and above_free) or (behind_below_diag_blocked and below_free):
                return (x, y), intermediate_nodes
        # Vertical
        elif dx == 0 and dy != 0:
            behind_right_diag_blocked = not traversable(grid, x + 1, y - dy)
            right_free = traversable(grid, x + 1, y)
            behind_left_diag_blocked = not traversable(grid, x - 1, y - dy)
            left_free = traversable(grid, x - 1, y)
            if (behind_right_diag_blocked and right_free) or (behind_left_diag_blocked and left_free):
                return (x, y), intermediate_nodes
        # Diagonal
        else:
            # Check corner cutting
            if not traversable(grid, x - dx, y) or not traversable(grid, x, y - dy):
                return None, intermediate_nodes
            
            h_jump, h_nodes = jump(grid, (x, y), dx, 0, goal)
            v_jump, v_nodes = jump(grid, (x, y), 0, dy, goal)
            if h_jump or v_jump:
                intermediate_nodes.extend(h_nodes[:-1])
                intermediate_nodes.extend(v_nodes[:-1])
                return (x, y), intermediate_nodes

def jps_neighbours(grid: np.ndarray, current: Coord, goal: Coord) -> Iterator[tuple[Coord, float, list[Coord]]]:
    """
    Get the jump point successors (neighbours) of the current point.

    Args:
        grid: the grid we are exploring
        current: the current point
        goal: the goal point

    Returns:
        the jump points along with their heuristic
    """
    for dx, dy in DIRECTIONS:
        next_cell, intermediate_nodes = jump(grid, current, dx, dy, goal)
        if next_cell:
            yield next_cell, heuristic(current, next_cell), intermediate_nodes 

##################################################
# Wrappers to call each algorithm
##################################################

def a_star_explore(grid, start, goal):
    return generic_search(grid, start, goal, a_star_neighbours)

def jps_explore(grid, start, goal):
    return generic_search(grid, start, goal, jps_neighbours)

##################################################
# Main script execution
##################################################

if __name__ == "__main__":
    grid = load_map("CatwalkAlley.map")
    grid = grid.astype(np.int8) # Save on some memory
    start = (278,90)
    goal = (86,130)
    padding = min(max(abs(start[0]) - abs(goal[0]), abs(start[1]) - abs(goal[1])), 20)
    grid, start, goal = trim_grid(grid, start, goal, pad=padding)

    t0 = time.time()
    path_jps, frames_jps = jps_explore(grid, start, goal)
    t1 = time.time()
    jps_time = t1-t0
    print(f"JPS Took {jps_time:.5f}s")

    t0 = time.time()
    path_astar, frames_astar = a_star_explore(grid, start, goal)
    t1 = time.time()
    astar_time = t1-t0
    print(f"A* Took {astar_time:.5f}s")

    print(f"JPS is {astar_time/jps_time:.5f}x quicker")
    print(f"JPS path length: {path_length(path_jps)}")
    print(f"A* path length: {path_length(path_astar)}")

    # 0 = free, 1 = shallow water, 2 = tree, 3 = water, 4 = oob, 5 = start, 6 = goal, 7 = robot, 8 = explored, 9 = final path
    cmap = colors.ListedColormap(["white", "lightblue", "limegreen", "blue", "black", "green", "magenta", "red", "gray", "orange"])
    bounds = [0,1,2,3,4,5,6,7,8,9,10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    # initial display copies
    display_jps = grid.copy()
    display_astar = grid.copy()

    display_jps[start[1], start[0]] = 5
    display_jps[goal[1], goal[0]] = 6
    display_astar[start[1], start[0]] = 5
    display_astar[goal[1], goal[0]] = 6

    im_jps = ax1.imshow(display_jps, cmap=cmap, norm=norm, origin='upper')
    ax1.set_title("JPS Exploration")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    im_astar = ax2.imshow(display_astar, cmap=cmap, norm=norm, origin='upper')
    ax2.set_title("A* Exploration")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    x_jps, y_jps = start
    x_astar, y_astar = start

    def update(frame_index: int):
        global x_jps, y_jps, x_astar, y_astar # Keep in scope
        # JPS frame
        if frame_index < len(frames_jps):
            current, _, _ = frames_jps[frame_index]
            display_jps[y_jps, x_jps] = 8 # Using previous x, y values set to explored
            x_jps, y_jps = current
            display_jps[y_jps, x_jps] = 7 # New explored point
            if path_jps and frame_index == len(frames_jps) - 1:
                for i in range(len(path_jps)-1):
                    x1, y1 = path_jps[i]
                    x2, y2 = path_jps[i+1]
                    ax1.plot([x1, x2],[y1, y2], color="orange", linewidth=1)
            mark_point(display_jps, start, 5, size=3)
            mark_point(display_jps, goal, 6, size=3)
            im_jps.set_data(display_jps)
            ax1.set_title(f"JPS Exploration: {frame_index + 1} nodes")

        # A* frame
        if frame_index < len(frames_astar):
            current, _, _ = frames_astar[frame_index]
            display_astar[y_astar, x_astar] = 8 # Using previous x, y values set to explored
            x_astar, y_astar = current
            display_astar[y_astar, x_astar] = 7 # New explored point
            if path_astar and frame_index == len(frames_astar) - 1:
                for i in range(len(path_astar)-1):
                    x1, y1 = path_astar[i]
                    x2, y2 = path_astar[i+1]
                    ax2.plot([x1, x2],[y1, y2], color="orange", linewidth=1)
            mark_point(display_astar, start, 5, size=3)
            mark_point(display_astar, goal, 6, size=3)
            im_astar.set_data(display_astar)
            ax2.set_title(f"A* Exploration: {frame_index + 1} nodes")

        return [im_jps, im_astar]

    max_frames = max(len(frames_jps), len(frames_astar))
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=1000*(1/60), blit=False, repeat=False)
    plt.show()

    final_jps = build_final_display(grid, start, goal, frames_jps)
    final_astar = build_final_display(grid, start, goal, frames_astar)
    if not os.path.exists("figs"):
        os.makedirs("figs")
    save_final(final_jps, path_jps, f"JPS Exploration: {len(frames_jps[-1][1])}", "figs/jps.png")
    save_final(final_astar, path_astar, f"A* Exploration: {len(frames_astar[-1][1])}", "figs/astar.png")
