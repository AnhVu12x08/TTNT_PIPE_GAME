"""
Pipeline Connection Game

Developed by: Team ZLO
Members:
- Trương Anh Vũ - 2033225898
- Phạm Hoàng Quân - 2033223962
- Kiều Hoàng Thái - 2033224647

Description:
This Python script implements a pipeline connection game where players rotate pipes on a grid to establish a flow
from point A to point B. The game uses the Hill-Climbing algorithm to solve graph-based puzzles.

Date: November 30, 2024
Version: 1.0
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import sys
import os
import copy

# Grid size
GRID_SIZE = 6

# Pipe types and rotations
PIPE_TYPES = ["straight", "bent"]
ROTATIONS = [0, 90, 180, 270]


# Define connections based on pipe type and rotation
def get_pipe_connections(pipe_type, rotation):
    """
    Determines the connected ends of a pipe based on its type and rotation.

    :param pipe_type: The type of pipe ("straight" or "bent").
    :param rotation: The rotation of the pipe in degrees (0, 90, 180, 270).
    :return: A list of strings representing the connected ends ("top", "bottom", "left", "right"). Returns an empty list if the pipe type or rotation is invalid.
    """
    connections = []  # Initialize the list of connections
    if pipe_type == "straight":
        if rotation == 0 or rotation == 180:
            connections = ["top", "bottom"]  # Straight pipe, vertical
        elif rotation == 90 or rotation == 270:
            connections = ["left", "right"]  # Straight pipe, horizontal
    elif pipe_type == "bent":
        if rotation == 0:
            connections = ["top", "right"]  # Bent pipe, top-right
        elif rotation == 90:
            connections = ["right", "bottom"]  # Bent pipe, right-bottom
        elif rotation == 180:
            connections = ["bottom", "left"]  # Bent pipe, bottom-left
        elif rotation == 270:
            connections = ["left", "top"]  # Bent pipe, left-top
    return connections


# Check if two pipes are connected
def are_pipes_connected(pipe1, pipe2, direction):
    """
    :param pipe1: A dictionary representing the first pipe (contains "connections" key).
    :param pipe2: A dictionary representing the second pipe (contains "connections" key).
    :param direction: The direction to check the connections ("top", "bottom", "left", "right")
    :return: True if the connections are connected, False otherwise
    """
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    return direction in pipe1["connections"] and opposite[direction] in pipe2["connections"]


class StartMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Pipe Connection Game")
        self.width = 700
        self.height = 700
        self.root.geometry(f"{self.width}x{self.height}")
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root, bg="#e0f2f7")
        frame.place(relx=0.5, rely=0.5, anchor="center")

        self.welcome_label = tk.Label(frame, text="Welcome Plumber!", font=("Arial", 28, "bold"), fg="#2196f3",
                                      bg="#e0f2f7")
        self.welcome_label.pack(pady=30)

        self.start_button = tk.Button(frame, text="Start", command=self.open_level_selection, font=("Arial", 18),
                                      bg="#4caf50", fg="white", relief="raised", bd=3)
        self.start_button.pack()

    def open_level_selection(self):
        self.root.destroy()
        level_root = tk.Tk()
        LevelSelection(level_root)
        level_root.mainloop()


class LevelSelection:
    def __init__(self, root):
        self.root = root
        self.root.title("Select Level")
        self.root.geometry("700x500")
        self.create_level_buttons()

    def create_level_buttons(self):
        for i in range(1, 5):
            level_button = tk.Button(self.root, text=f"Level {i}", command=lambda level=i: self.start_game(level),
                                     font=("Arial", 14))
            level_button.pack(pady=10)

    def start_game(self, level):
        global GRID_SIZE
        if level == 1:
            GRID_SIZE = 5
        elif level == 2:
            GRID_SIZE = 6
        elif level == 3:
            GRID_SIZE = 7
        elif level == 4:
            GRID_SIZE = 8

        self.root.destroy()
        game_root = tk.Tk()
        game = PipeGame(game_root)
        game_root.mainloop()


class PipeGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Pipe Connection Game")
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()

        self.cell_size = 500 // GRID_SIZE
        self.start = (0, 0)
        self.end = (GRID_SIZE - 1, GRID_SIZE - 1)

        self.images = self.load_images()

        self.create_grid()
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_click)
        self.create_solve_button()

        self.selected_algorithm = None

        self.create_menu()

    def resource_path(self, relative_path):
        """
        Gets the absolute path to a resource, works for both bundled and unbundled applications.

        :param relative_path: The path to the resource relative to the script's location or the bundle's location.
        :return: The absolute path to the resource.
        """
        try:
            base_path = sys._MEIPASS  # PyInstaller creates a temporary directory at runtime
        except Exception:
            base_path = os.path.abspath(".")  # For running the script directly
        return os.path.join(base_path, relative_path)

    def load_images(self):
        """
        Loads and resizes pipe images from the "images" folder.

        :return: A dictionary where keys are (pipe_type, rotation) tuples and values are ImageTk.PhotoImage objects. Returns None if any image file is not found.
        """
        images = {}
        for pipe_type in PIPE_TYPES:
            for rotation in ROTATIONS:
                filename = f"{pipe_type}_{rotation}.png"
                filepath = self.resource_path(os.path.join("images", filename))

                try:
                    image = Image.open(filepath)
                    image = image.resize((self.cell_size, self.cell_size), Image.Resampling.LANCZOS)  # Use LANCZOS
                    images[(pipe_type, rotation)] = ImageTk.PhotoImage(image)
                except FileNotFoundError:
                    print(f"Error: Image file not found: {filepath}")
                    return

        return images

    def create_grid(self):
        """
        Creates a solvable grid of pipes, ensuring the start and end pipes are straight and correctly oriented.  Uses
        recursion to ensure a solvable grid is generated.

        :return: None
        """
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Place start and end pipes (straight, specific rotations)
        self.grid[self.start[0]][self.start[1]] = {"type": "straight", "rotation": 0,
                                                   "connections": get_pipe_connections("straight", 0)}
        self.grid[self.end[0]][self.end[1]] = {"type": "straight", "rotation": 180,
                                               "connections": get_pipe_connections("straight", 180)}

        # Generate path and fill with pipes
        path = self.generate_valid_path()
        self.fill_path_with_pipes(path)

        # Fill remaining cells
        self.fill_remaining_cells()
        if self.check_win() == False:  # Check if the grid is solvable. If not, regenerate the grid.
            self.create_grid()
        self.draw_grid()  # Draw the grid on the canvas
        self.rotate_all_pipes() # Randomly rotate all pipes (excluding start and end)

    def generate_valid_path(self):
        """
        Generates a valid path from the start to the end cell using Depth-First Search (DFS).  The path is randomized
        by shuffling the move order.

        :return: A list of (row, column) tuples representing the path from start to end. Returns an empty list if no
        path is found (shouldn't happen with a properly sized grid).
        """
        path = []
        visited = set()

        def dfs(current):
            if current == self.end:
                return True

            visited.add(current)
            path.append(current)

            row, col = current
            # Possible moves: right, down, left, up
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(moves)  # Randomize path generation

            for dr, dc in moves:  # Iterate through the shuffled moves
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)

                if (0 <= new_row < GRID_SIZE and
                        0 <= new_col < GRID_SIZE and
                        new_pos not in visited):
                    if dfs(new_pos): # Recursive call to explore the next cell
                        return True

            path.pop() # Backtrack: If no path is found from the current cell, remove it from the path
            return False

        dfs(self.start) # Start the DFS from the start cell
        return path

    def fill_path_with_pipes(self, path):
        for i in range(len(path)):
            current = path[i]
            row, col = current
            connections = []
            if i > 0:
                prev_row, prev_col = path[i - 1]
                if prev_row < row:
                    connections.append("top")
                elif prev_row > row:
                    connections.append("bottom")
                elif prev_col < col:
                    connections.append("left")
                elif prev_col > col:
                    connections.append("right")
            if i < len(path) - 1:
                next_row, next_col = path[i + 1]
                if next_row < row:
                    connections.append("top")
                elif next_row > row:
                    connections.append("bottom")
                elif next_col < col:
                    connections.append("left")
                elif next_col > col:
                    connections.append("right")

            if not connections:
                self.grid[row][col] = {"type": "straight", "rotation": 0,
                                       "connections": get_pipe_connections("straight", 0)}
            else:
                pipe_type = "bent" if len(set(connections)) == 2 and \
                                      ("top" in connections and "right" in connections or
                                       "right" in connections and "bottom" in connections or
                                       "bottom" in connections and "left" in connections or
                                       "left" in connections and "top" in connections) else "straight"

                rotation = self.get_appropriate_rotation(pipe_type, connections)
                self.grid[row][col] = {"type": pipe_type, "rotation": rotation,
                                       "connections": get_pipe_connections(pipe_type, rotation)}

    def get_appropriate_rotation(self, pipe_type, connections):
        """
        Determines the appropriate rotation for a pipe based on its type and the required connections.

        :param pipe_type: The type of the pipe ("straight" or "bent").
        :param connections: A list of strings
        indicating the required connections ("top", "bottom", "left", "right").
        :return: The appropriate rotation in
        degrees (0, 90, 180, 270). Returns -1 if an invalid pipe type or connection is provided.
        """
        if pipe_type == "straight":
            if "top" in connections or "bottom" in connections:
                return 0
            return 90
        else:  # bent pipe
            if "top" in connections and "right" in connections:
                return 0
            if "right" in connections and "bottom" in connections:
                return 90
            if "bottom" in connections and "left" in connections:
                return 180
            return 270

    def fill_remaining_cells(self):
        """Fills remaining empty cells with random pipes"""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] is None:
                    pipe_type = random.choice(PIPE_TYPES)
                    rotation = random.choice(ROTATIONS)
                    self.grid[row][col] = {
                        "type": pipe_type,
                        "rotation": rotation,
                        "connections": get_pipe_connections(pipe_type, rotation)
                    }

    # rotate all pipe in gird
    def rotate_all_pipes(self):
        """
        Randomly rotates all pipes in the grid except for the start and end pipes.

        :return: None
        """
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if (row, col) == self.start or (row, col) == self.end:
                    continue  # Do not rotate the start or end pipes

                # Rotate pipe to a random rotation
                pipe = self.grid[row][col]
                pipe["rotation"] = random.choice(ROTATIONS)
                pipe["connections"] = get_pipe_connections(pipe["type"], pipe["rotation"])
        self.draw_grid()

    def draw_grid(self):
        """Draw the pipes on the canvas"""
        self.canvas.delete("all")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                pipe = self.grid[row][col]
                img = self.images[(pipe["type"], pipe["rotation"])]

                self.canvas.create_image(x1, y1, anchor="nw", image=img)

                if (row, col) == self.start or (row, col) == self.end:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def rotate_pipe(self, row, col):
        """
        Rotates the pipe at the specified row and column by 90 degrees clockwise.  Start and end pipes are not rotated.

        :param row: The row index of the pipe.
        :param col: The column index of the pipe.
        :return: None
        """
        if (row, col) == self.start or (row, col) == self.end:
            return

        pipe = self.grid[row][col]
        pipe["rotation"] = (pipe["rotation"] + 90) % 360  # Rotate the pipe by 90 degrees clockwise
        pipe["connections"] = get_pipe_connections(pipe["type"], pipe["rotation"])  # Update the connections based on
        # the new rotation

    def on_click(self, event):
        """
        Handle click events
        :param event: The mouse click event.
        :return: None
        """
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE: # Check if the click is within the grid
            self.rotate_pipe(row, col)
            self.draw_grid() # Redraw the grid to show the change

            if self.check_win():
                messagebox.showinfo("Game Over", "You Win!")
                self.reset_game()

    def check_win(self):
        """Check if all pipes are connected from start to end"""
        visited = set()
        stack = [self.start]

        while stack:
            current = stack.pop()
            if current == self.end:
                return True

            if current in visited:
                continue

            visited.add(current)
            row, col = current

            neighbors = {
                "top": (row - 1, col),
                "bottom": (row + 1, col),
                "left": (row, col - 1),
                "right": (row, col + 1)
            }

            for direction, (n_row, n_col) in neighbors.items():
                if 0 <= n_row < GRID_SIZE and 0 <= n_col < GRID_SIZE:
                    neighbor_pipe = self.grid[n_row][n_col]
                    if are_pipes_connected(self.grid[row][col], neighbor_pipe, direction):
                        stack.append((n_row, n_col))

        return False

    def create_solve_button(self):
        """
        Creates and packs the "Solve" button that triggers the puzzle solving algorithm.

        :return: None
        """
        solve_button = tk.Button(self.root, text="Solve", command=self.solve_puzzle)  # Create the button
        solve_button.pack()  # Pack the button into the layout

    def solve_puzzle(self):
        """Solves the puzzle based on the selected algorithm."""
        if self.selected_algorithm == "hill_climbing":
            self.temp_function1()
        elif self.selected_algorithm == "DFS":
            self.temp_function2()
        else:
            messagebox.showinfo("Error", "Please select an algorithm from the 'Algorithm' menu.")

    def temp_func(self):
        messagebox.showinfo(title="hehe", message=f"Debuggggggggggggggggggggg")

    def reset_game(self):
        """Resets the game to a new random configuration."""
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.create_grid()
        self.draw_grid()

    def temp_function1(self):
        """
        Solves the pipe puzzle using a stochastic hill-climbing algorithm.

        :return: None. Modifies self.grid directly if a solution is found.
        """
        current_state = copy.deepcopy(self.grid)  # Create a copy of the current grid state to avoid modifying the original
        current_score = self.calculate_score(current_state)  # Calculate the initial score of the copied grid

        iterations = 0
        max_iterations = 50000

        while current_score < 100 and iterations < max_iterations:
            improved = False

            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) != self.start and (r, c) != self.end:
                        best_rotation = current_state[r][c]["rotation"]
                        best_score_for_pipe = current_score

                        for _ in range(3):  # Try all 3 other rotations for this ONE pipe
                            current_state[r][c]["rotation"] = (current_state[r][c]["rotation"] + 90) % 360
                            current_state[r][c]["connections"] = get_pipe_connections(current_state[r][c]["type"],
                                                                                      current_state[r][c]["rotation"])
                            score = self.calculate_score(current_state)

                            if score > best_score_for_pipe:
                                best_score_for_pipe = score
                                best_rotation = current_state[r][c]["rotation"]
                                improved = True  # Flag to indicate an improvement was made

                        # Set the best rotation found for the current pipe
                        current_state[r][c]["rotation"] = best_rotation
                        current_state[r][c]["connections"] = get_pipe_connections(current_state[r][c]["type"],
                                                                                  best_rotation)
                        current_score = best_score_for_pipe  # Update the overall score

            if not improved:
                self.rotate_some_pipes(current_state, 5)  # Rotate 5 random pipes
                current_score = self.calculate_score(current_state)

            iterations += 1

        self.grid = current_state
        self.draw_grid()

        if current_score == 100:
            messagebox.showinfo("Game Over", "Hill Climbing Solved!")
        else:
            messagebox.showinfo("Game Over", f"Hill Climbing did not find a solution after {iterations} iterations.")

    def rotate_some_pipes(self, state, num_rotations=1):
        """
            Rotates a specified number of random pipes in the given grid state.  Start and end pipes are not rotated.

            :param state: The grid state (list of lists of dictionaries) to modify.
            :param num_rotations: The number of pipes to rotate. Defaults to 1.
            :return: None. Modifies the input 'state' directly.
            """
        rotatable_pipes = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if
                           (r, c) != self.start and (r, c) != self.end]
        pipes_to_rotate = random.sample(rotatable_pipes, min(num_rotations, len(rotatable_pipes)))

        for r, c in pipes_to_rotate:
            state[r][c]["rotation"] = random.choice(ROTATIONS)
            state[r][c]["connections"] = get_pipe_connections(state[r][c]["type"], state[r][c]["rotation"])

    def calculate_score(self, state):
        """
        Calculates a score representing how close the given pipe configuration is to being solved. Uses Depth-First
        Search (DFS).

        :param state: The grid state (list of lists of dictionaries) to evaluate.
        :return: An integer score between 0 and 100, inclusive.  100 indicates a solved puzzle.
        """
        visited = set()
        stack = [self.start]
        count = 0
        while stack:
            current = stack.pop()
            if current == self.end:
                return 100

            if current in visited:
                continue
            visited.add(current)
            row, col = current

            neighbors = {
                "top": (row - 1, col),
                "bottom": (row + 1, col),
                "left": (row, col - 1),
                "right": (row, col + 1)
            }

            for direction, (n_row, n_col) in neighbors.items():
                if 0 <= n_row < GRID_SIZE and 0 <= n_col < GRID_SIZE:
                    neighbor_pipe = state[n_row][n_col]
                    if are_pipes_connected(state[row][col], neighbor_pipe, direction):
                        stack.append((n_row, n_col))
                        count += 1

        return count

    def temp_function2(self):
        """Depth-First Search (DFS) algorithm to solve the pipe puzzle."""

        def dfs(grid, current_pos, visited):
            if current_pos == self.end:
                return True, grid

            visited.add(current_pos)

            row, col = current_pos
            neighbors = {
                "top": (row - 1, col),
                "bottom": (row + 1, col),
                "left": (row, col - 1),
                "right": (row, col + 1)
            }

            for direction, (n_row, n_col) in neighbors.items():
                if (0 <= n_row < GRID_SIZE and 0 <= n_col < GRID_SIZE and
                        (n_row, n_col) not in visited):

                    for _ in range(4):  # Try all rotations of the neighbor
                        if (n_row, n_col) != self.start and (n_row, n_col) != self.end:  # Do not rotate start/end
                            grid[n_row][n_col]["rotation"] = (grid[n_row][n_col]["rotation"] + 90) % 360
                            grid[n_row][n_col]["connections"] = get_pipe_connections(grid[n_row][n_col]["type"],
                                                                                     grid[n_row][n_col]["rotation"])

                        if are_pipes_connected(grid[row][col], grid[n_row][n_col], direction):
                            solved, solution = dfs(copy.deepcopy(grid), (n_row, n_col), visited.copy())
                            if solved:
                                return True, solution

            return False, None

        solved, solution_grid = dfs(copy.deepcopy(self.grid), self.start, set())

        if solved:
            self.grid = solution_grid
            self.draw_grid()
            messagebox.showinfo("Game Over", "DFS Solved!")

        else:
            messagebox.showinfo("Game Over", "DFS did not find a solution.")

    def create_menu(self):
        """Create a menu bar with reset and temp function options."""

        menubar = tk.Menu(self.root)
        settingmenu = tk.Menu(menubar, tearoff=0)
        settingmenu.add_command(label="Reset", command=self.reset_game)
        settingmenu.add_separator()
        settingmenu.add_command(label="Exit", command=self.return_to_level_selection)
        menubar.add_cascade(label="Setting", menu=settingmenu)

        AlgorMenu = tk.Menu(menubar, tearoff=0)
        AlgorMenu.add_command(label="Hill Climbing", command=lambda: self.set_algorithm("hill_climbing"))
        AlgorMenu.add_command(label="Depth-First Search (DFS)", command=lambda: self.set_algorithm("DFS"))
        menubar.add_cascade(label="Algorithm", menu=AlgorMenu)
        self.root.config(menu=menubar)

    def set_algorithm(self, algorithm_name):
        """Sets the selected algorithm."""
        self.selected_algorithm = algorithm_name
        print(f"Selected algorithm: {algorithm_name}")

    def return_to_level_selection(self):
        self.root.destroy()
        level_root = tk.Tk()
        LevelSelection(level_root)
        level_root.mainloop()


# Run the game
root = tk.Tk()
start_menu = StartMenu(root)

root.mainloop()


