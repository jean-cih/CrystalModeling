import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, Button


class InteractiveCrystalViewer:
    def __init__(self):
        self.Lx, self.Ly, self.Lz = 2, 2, 2
        self.Nx, self.Ny, self.Nz = 35, 35, 35
        self.wavelength = 515e-3
        self.n = 1.5

        self.lights = [
            {'amplitude': 1, 'theta': 40, 'phi': 0},
            {'amplitude': 1, 'theta': 0, 'phi': 0},
            {'amplitude': 1, 'theta': 20, 'phi': 120},
            {'amplitude': 1, 'theta': 0, 'phi': 240}
        ]

        self.disk_color = (12/255, 128/255, 128/255)
        self.edge_color = (20/255, 200/255, 200/255)

        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.z = np.linspace(0, self.Lz, self.Nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        self.intensity = self.calculate_interference()

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.setup_ui()

        self.update_visualization()

    def calculate_interference(self):
        k = 2 * np.pi / (self.wavelength / self.n)
        field = np.zeros_like(self.X, dtype=complex)

        for light in self.lights:
            theta = np.deg2rad(light['theta'])
            phi = np.deg2rad(light['phi'])
            kx = k * np.sin(theta) * np.cos(phi)
            ky = k * np.sin(theta) * np.sin(phi)
            kz = k * np.cos(theta)
            field += light['amplitude'] * np.exp(1j * (kx*self.X + ky*self.Y + kz*self.Z))

        return np.abs(field)**2

    def setup_ui(self):
        ax_elev = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_azim = plt.axes([0.25, 0.05, 0.65, 0.03])

        self.slider_elev = Slider(ax_elev, 'Elevation', -180, 180, valinit=30)
        self.slider_azim = Slider(ax_azim, 'Azimuth', -180, 180, valinit=-45)

        resetax = plt.axes([0.8, 0.15, 0.1, 0.04])
        self.button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

        self.slider_elev.on_changed(self.update_view)
        self.slider_azim.on_changed(self.update_view)
        self.button.on_clicked(self.reset_view)

    def update_view(self, val):
        self.ax.view_init(elev=self.slider_elev.val, azim=self.slider_azim.val)
        self.fig.canvas.draw_idle()

    def reset_view(self, event):
        self.slider_elev.reset()
        self.slider_azim.reset()

    def update_visualization(self, threshold=0.5):
        self.ax.clear()

        try:
            verts, faces, _, _ = marching_cubes(self.intensity, level=threshold*np.max(self.intensity))

            verts[:, 0] *= self.Lx / self.Nx
            verts[:, 1] *= self.Ly / self.Ny
            verts[:, 2] *= self.Lz / self.Nz

            mesh = Poly3DCollection(verts[faces], alpha=0.8)
            mesh.set_facecolor(self.disk_color)
            mesh.set_edgecolor(self.edge_color)
            self.ax.add_collection3d(mesh)

            self.ax.set_xlim(0, self.Lx)
            self.ax.set_ylim(0, self.Ly)
            self.ax.set_zlim(0, self.Lz)
            self.ax.set_xlabel('X (мкм)')
            self.ax.set_ylabel('Y (мкм)')
            self.ax.set_zlabel('Z (мкм)')
            self.ax.set_title('Интерактивная 3D модель кристаллической структуры')

            self.ax.view_init(elev=30, azim=-45)

        except Exception as e:
            print(f"Ошибка визуализации: {e}")

        plt.tight_layout()

    def show(self):
        """Отображение интерактивного окна"""
        plt.show()


if __name__ == '__main__':
    viewer = InteractiveCrystalViewer()
    viewer.show()
