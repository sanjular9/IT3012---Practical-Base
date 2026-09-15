from collections import deque
import heapq
import math


class SearchAgent:

    def __init__(self):
        self.plan = []
        self.active_algo = "BFS"

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt(
            (pos[0] - goal[0]) ** 2 +
            (pos[1] - goal[1]) ** 2
        )

    def get_neighbors(self, state, walls, width, height):
        x, y = state

        moves = [
            ("Up", (x, y + 1)),
            ("Down", (x, y - 1)),
            ("Left", (x - 1, y)),
            ("Right", (x + 1, y))
        ]

        for action, next_state in moves:
            nx, ny = next_state

            if 0 <= nx < width and 0 <= ny < height and next_state not in walls:
                yield action, next_state

    def build_path(self, parent, state):
        path = []

        while parent[state][0] is not None:
            previous, action = parent[state]
            path.append(action)
            state = previous

        path.reverse()
        return path

    def bfs(self, start, goal, walls, width, height):
        frontier = deque([start])
        reached = {start}
        parent = {start: (None, None)}

        while frontier:
            current = frontier.popleft()

            if current == goal:
                return self.build_path(parent, current)

            for action, next_state in self.get_neighbors(
                current, walls, width, height
            ):
                if next_state not in reached:
                    reached.add(next_state)
                    parent[next_state] = (current, action)
                    frontier.append(next_state)

        return []

    def dfs(self, start, goal, walls, width, height):
        frontier = [start]
        reached = {start}
        parent = {start: (None, None)}

        while frontier:
            current = frontier.pop()

            if current == goal:
                return self.build_path(parent, current)

            for action, next_state in self.get_neighbors(
                current, walls, width, height
            ):
                if next_state not in reached:
                    reached.add(next_state)
                    parent[next_state] = (current, action)
                    frontier.append(next_state)

        return []

    def ucs(self, start, goal, walls, width, height):
        frontier = [(0, start)]
        reached = {start}
        costs = {start: 0}
        parent = {start: (None, None)}

        while frontier:
            cost, current = heapq.heappop(frontier)

            if current == goal:
                return self.build_path(parent, current)

            for action, next_state in self.get_neighbors(
                current, walls, width, height
            ):
                new_cost = cost + 1

                if next_state not in reached or new_cost < costs.get(next_state, float("inf")):
                    reached.add(next_state)
                    costs[next_state] = new_cost
                    parent[next_state] = (current, action)
                    heapq.heappush(frontier, (new_cost, next_state))

        return []

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type="manhattan"):
        width, height = grid_size
        frontier = []

        if heuristic_type == "euclidean":
            h = self.euclidean_distance(start_pos, goal_pos)
        else:
            h = self.manhattan_distance(start_pos, goal_pos)

        heapq.heappush(frontier, (h, 0, start_pos, []))
        reached_states = set()

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            for action, next_pos in self.get_neighbors(
                current_pos, walls, width, height
            ):
                if next_pos not in reached_states:
                    new_g = g_cost + 1

                    if heuristic_type == "euclidean":
                        new_h = self.euclidean_distance(next_pos, goal_pos)
                    else:
                        new_h = self.manhattan_distance(next_pos, goal_pos)

                    new_f = new_g + new_h

                    heapq.heappush(
                        frontier,
                        (new_f, new_g, next_pos, path_taken + [action])
                    )

        return []

    def find_plan(self, percept):
        start = tuple(percept["agent_pos"])
        food = [tuple(f) for f in percept["remaining_food"]]

        if not food:
            return []

        width, height = percept["grid_size"]
        walls = {tuple(w) for w in percept["walls"]}

        # Search for the nearest food pellet.
        goal = min(
            food,
            key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1])
        )

        if self.active_algo == "BFS":
            return self.bfs(start, goal, walls, width, height)

        if self.active_algo == "DFS":
            return self.dfs(start, goal, walls, width, height)

        if self.active_algo == "UCS":
            return self.ucs(start, goal, walls, width, height)

        if self.active_algo == "AStar":
            return self.astar_search(
                start,
                goal,
                walls,
                (width, height),
                "manhattan"
            )

        return []

    def sense_and_act(self, percept):
        if not self.plan:
            self.plan = self.find_plan(percept)

        if self.plan:
            return self.plan.pop(0)

        return "Stay"


