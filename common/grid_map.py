import numpy as np
import matplotlib.pyplot as plt


class GridMap:
    def __init__(self, width, height, xy_resolution=2.0, heading_resolution=np.deg2rad(15.0)):
        self.xy_res = xy_resolution  # meters
        self.heading_res = heading_resolution  # radians
        self.motion_res = 0.1  # meters

        self._map = np.zeros((int(width / self.xy_res), int(height / self.xy_res)))

    def plot_grid_map(self):
        plt.imshow(self._map, origin='lower', cmap='gray')
        plt.show()


def main():
    grid_map = GridMap(100, 100)
    grid_map.plot_grid_map()


if __name__ == '__main__':
    main()


