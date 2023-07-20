from gridworld.tile_types import TileType
import torch
import random

class World:
    def __init__(self, grid_size: int, instances: dict[int, int] = {}):
        self.grid_size = grid_size
        self.grid = torch.full((self.grid_size, self.grid_size), fill_value=TileType.empty.value, dtype=torch.int32)
        self.reset(instances)

    def reset(self, instances: dict[int, int]) -> None:
        self._clear_grid()
        self._place_instances(instances)

    def _clear_grid(self):
        self.grid.fill_(TileType.empty.value)

    def _place_instances(self, instances: dict[int, int]) -> None:
        for tile_type, count in instances.items():
            for _ in range(count):
                x, y = self.get_random_coordinates()
                self.grid[y, x] = tile_type

    def is_complete(self) -> bool:
        return False

    def in_bounds(self, x, y) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_tile(self, x: int, y: int) -> int:
        if self.in_bounds(x, y):
            return int(self.grid[y, x].item())
        return TileType.empty.value
    
    def get_random_coordinates(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        return x, y

    def get_visual_field_at_point(self, x: int, y: int, vision_range: int) -> torch.Tensor:
        """Returns the visual field at the given point with the given vision range."""
        visual_field = torch.full((2 * vision_range + 1, 2 * vision_range + 1), fill_value=TileType.empty.value, dtype=torch.int32)
        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                tile = self.get_tile(x + dx, y + dy)
                visual_field[dy + vision_range, dx + vision_range] = tile
        return visual_field

    def get_visual_field_at_agent(self, agent) -> torch.Tensor:
        """Gets the visual field from the given agent."""
        return self.get_visual_field_at_point(agent.x, agent.y, agent.vision_range)


class LoggerTrainingWorld(World):
    def __init__(self, grid_size: int = 10, instances: dict[int, int] = { TileType.tree.value: 20, TileType.log.value: 0 }):
        super().__init__(grid_size)
        self.reset(instances)

    def is_complete(self) -> bool:
        return torch.count_nonzero(self.grid == TileType.tree.value).item() == 0
