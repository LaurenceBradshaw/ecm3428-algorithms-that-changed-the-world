import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from matplotlib import colors
import time
import math
import os

DIRECTIONS = [(-1,-1),(1,1),(-1,1),(1,-1),(-1,0),(1,0),(0,-1),(0,1)]
CORNER_CUTTING=False

##################################################
# Utilities
##################################################

def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        length += math.sqrt(dx**2+dy**2)
    return length

def load_map(path):
    height = None
    width = None

    # First pass: read metadata and then read the map
    with open(path, "r") as file_handle:
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
                    grid[row_index, col_index] = 0
                elif ch == "S":
                    grid[row_index, col_index] = 1
                elif ch == "T":
                    grid[row_index, col_index] = 2
                elif ch == "W":
                    grid[row_index, col_index] = 3
                elif ch == "@":
                    grid[row_index, col_index] = 4
                else:
                    raise ValueError(f"Unknown map symbol: {ch}")
                col_index += 1

            row_index += 1

        if row_index != height:
            raise ValueError("Number of map rows does not match declared height")

    return grid

def trim_grid(grid, start, goal, pad=0):
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

def traversable(grid, x, y):
    out_of_bounds = not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0])
    if out_of_bounds:
        return False

    val = grid[y, x]
    if val == 2 or val == 3 or val == 4:
        return False
    else:
        return True
    
def mark_point(display, point, value, size=3):
    c, r = point
    rows, cols = display.shape
    half = size // 2
    r_min = max(r - half, 0)
    r_max = min(r + half + 1, rows)
    c_min = max(c - half, 0)
    c_max = min(c + half + 1, cols)
    display[r_min:r_max, c_min:c_max] = value
    
def build_final_display(grid, start, goal, frames):
    _, explored_set, _, _ = frames[-1]

    display = grid.copy()

    for x, y in explored_set:
        if display[y, x] == 0:
            display[y, x] = 8

    mark_point(display, start, 5, size=3)
    mark_point(display, goal, 6, size=3)

    return display

def save_final(display, path, title, filename):
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

def generic_search(grid, start, goal, neighbour_function, corner_cutting=False):
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
        frames.append((current, explored.copy(), came_from.copy(), g_score.copy()))

        # Reconstruct path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1], frames

        for neighbour, cost, intermediate_nodes in neighbour_function(grid, current, goal, corner_cutting=corner_cutting):
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

def a_star_neighbours(grid, current, goal, corner_cutting=False):
    for dx, dy in DIRECTIONS:
        nx, ny = current[0] + dx, current[1] + dy
        if not traversable(grid, nx, ny):
            continue
        if not corner_cutting:
            # Check corner cutting
            if not traversable(grid, nx - dx, ny) or not traversable(grid, nx, ny - dy):
                continue
        yield (nx, ny), heuristic(current, (nx,ny)), []

def jump(grid, current, dx, dy, goal, corner_cutting=False):
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

        # Main difference between cutting corners and not is where we evaluate a jump point.
        # If we cannot cut corners, we need the jump point to be beyond the obstacle to get around it.
        # If we can, then it is sufficent to put the jump point adjacent to the obstacle.

        if abs(dx) != abs(dy) and not traversable(grid, x + dx, y + dy):
            return None, intermediate_nodes
        
        if corner_cutting:
            # For horizontal and vertical:
            #   Check if the diagonals are free,
            #   if they are, then check one space backwards for an obstacle.
            #   If there is an obsticle, that space is a forced neighbour, so mark the current point as a jump point
            
            # Horizontal
            if dx != 0 and dy == 0:
                down_diag_free = traversable(grid, x + dx, y + 1)
                up_diag_free = traversable(grid, x + dx, y - 1)
                down_adjacent_blocked = not traversable(grid, x, y + 1)
                up_adjacent_blocked = not traversable(grid, x, y - 1)
                if (down_diag_free and down_adjacent_blocked) or (up_diag_free and up_adjacent_blocked):
                    return (x, y), intermediate_nodes
            # Vertical
            elif dx == 0 and dy != 0:
                right_diag_free = traversable(grid, x + 1, y + dy)
                left_diag_free = traversable(grid, x-1, y + dy)
                right_adjacent_blocked = not traversable(grid, x + 1, y)
                left_adjacent_blocked = not traversable(grid, x - 1, y)
                if (right_diag_free and right_adjacent_blocked) or (left_diag_free and left_adjacent_blocked):
                    return (x, y), intermediate_nodes
            # Diagonal
            else:
                diag_one_free = traversable(grid, x + dx, y + (-1*dy))
                adjacent_one_blocked = not traversable(grid, x, y + (-1*dy))
                diag_two_free = traversable(grid, x + (-1*dx), y + dy)
                adjacent_two_blocked = not traversable(grid, x + (-1*dx), y)
                if (diag_one_free and adjacent_one_blocked) or (diag_two_free and adjacent_two_blocked):
                    return (x,y), intermediate_nodes
                # Check horizontally and vertically in the direction of motion, ensuring we start from the current node
                # I.e., offset the +=dx and +=dy that will be done
                h_jump, h_nodes = jump(grid, (x, y), dx, 0, goal, corner_cutting)
                v_jump, v_nodes = jump(grid, (x, y), 0, dy, goal, corner_cutting)
                if h_jump or v_jump:
                    intermediate_nodes.extend(h_nodes[:-1])
                    intermediate_nodes.extend(v_nodes[:-1])
                    return (x, y), intermediate_nodes
        else:
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
                
                h_jump, h_nodes = jump(grid, (x, y), dx, 0, goal, corner_cutting)
                v_jump, v_nodes = jump(grid, (x, y), 0, dy, goal, corner_cutting)
                if h_jump or v_jump:
                    intermediate_nodes.extend(h_nodes[:-1])
                    intermediate_nodes.extend(v_nodes[:-1])
                    return (x, y), intermediate_nodes

def jps_neighbours(grid, current, goal, corner_cutting=False):
    for dx, dy in DIRECTIONS:
        next_cell, intermediate_nodes = jump(grid, current, dx, dy, goal, corner_cutting=corner_cutting)
        if next_cell:
            yield next_cell, heuristic(current, next_cell), intermediate_nodes 

##################################################
# Wrappers to call each algorithm
##################################################

def a_star_explore(grid, start, goal, corner_cutting=False):
    return generic_search(grid, start, goal, a_star_neighbours, corner_cutting=corner_cutting)

def jps_explore(grid, start, goal, corner_cutting=False):
    return generic_search(grid, start, goal, jps_neighbours, corner_cutting=corner_cutting)

##################################################
# Main script execution
##################################################

if __name__ == "__main__":
    grid = load_map("CatwalkAlley.map")
    start = (278,90)
    goal = (86,130)
    padding = min(max(abs(start[0]) - abs(goal[0]), abs(start[1]) - abs(goal[1])), 20)
    grid, start, goal = trim_grid(grid, start, goal, pad=padding)

    t0 = time.time()
    path_jps, frames_jps = jps_explore(grid, start, goal, corner_cutting=CORNER_CUTTING)
    t1 = time.time()
    jps_time = t1-t0
    print(f"JPS Took {jps_time:.5f}s")

    t0 = time.time()
    path_astar, frames_astar = a_star_explore(grid, start, goal, corner_cutting=CORNER_CUTTING)
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

    def update(frame_index):
        # JPS frame
        if frame_index < len(frames_jps):
            current, explored_set_jps, _, _ = frames_jps[frame_index]
            display_jps = grid.copy()
            for x, y in explored_set_jps:
                if display_jps[y, x] == 0:
                    display_jps[y, x] = 8
            if path_jps and frame_index == len(frames_jps) - 1:
                for i in range(len(path_jps)-1):
                    x1, y1 = path_jps[i]
                    x2, y2 = path_jps[i+1]
                    ax1.plot([x1, x2],[y1, y2], color="orange", linewidth=1)
            mark_point(display_jps, start, 5, size=3)
            mark_point(display_jps, goal, 6, size=3)
            display_jps[current[1], current[0]] = 7
            im_jps.set_data(display_jps)
            ax1.set_title(f"JPS Exploration: {len(explored_set_jps)} nodes")

        # A* frame
        if frame_index < len(frames_astar):
            current, explored_set_astar, _, _ = frames_astar[frame_index]
            display_astar = grid.copy()
            for x, y in explored_set_astar:
                if display_astar[y, x] == 0:
                    display_astar[y, x] = 8
            if path_astar and frame_index == len(frames_astar) - 1:
                for i in range(len(path_astar)-1):
                    x1, y1 = path_astar[i]
                    x2, y2 = path_astar[i+1]
                    ax2.plot([x1, x2],[y1, y2], color="orange", linewidth=1)
            mark_point(display_astar, start, 5, size=3)
            mark_point(display_astar, goal, 6, size=3)
            display_astar[current[1], current[0]] = 7
            im_astar.set_data(display_astar)
            ax2.set_title(f"A* Exploration: {len(explored_set_astar)} nodes")

        return [im_jps, im_astar]

    max_frames = max(len(frames_jps), len(frames_astar))
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=40, blit=False, repeat=False)
    plt.show()

    final_jps = build_final_display(grid, start, goal, frames_jps)
    final_astar = build_final_display(grid, start, goal, frames_astar)
    if not os.path.exists("figs"):
        os.makedirs("figs")
    save_final(final_jps, path_jps, f"JPS Exploration: {len(frames_jps[-1][1])}", "figs/jps.png")
    save_final(final_astar, path_astar, f"A* Exploration: {len(frames_astar[-1][1])}", "figs/astar.png")
