import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, Button, RadioButtons


class Crystal:
    def __init__(self, size=1, resolution=30, wavelength=515e-3, n=1.5):
        self.size = size
        self.resolution = resolution
        self.wavelength = wavelength
        self.n = n

        self.lights = [
            {'amplitude': 1, 'theta': 30, 'phi': 10},
            {'amplitude': 1, 'theta': 0, 'phi': 0},
            {'amplitude': 1, 'theta': 0, 'phi': 120},
            {'amplitude': 1, 'theta': 20, 'phi': 240}
        ]

        x = np.linspace(0, size, resolution)
        y = np.linspace(0, size, resolution)
        z = np.linspace(0, size, resolution)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')

        self.density = self.calculate_structure()

    def calculate_structure(self):
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


class Quasicrystal:
    def __init__(self, size=1, resolution=30, wavelength=0.5):
        self.size = size
        self.resolution = resolution
        self.wavelength = wavelength
        self.num_waves = 12
        self.phi = np.pi/2

        x = np.linspace(0, size, resolution)
        y = np.linspace(0, size, resolution)
        z = np.linspace(0, size, resolution)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')

        self.density = self.calculate_structure()

    def calculate_structure(self):
        density = np.zeros_like(self.X)
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.num_waves):
            theta = 2 * np.pi * i / self.num_waves
            phi = np.pi * i * golden_ratio
            kx = np.sin(theta) * np.cos(phi)
            ky = np.sin(theta) * np.sin(phi)
            kz = np.cos(theta)
            freq = 2 * np.pi / self.wavelength * (1 + 0.1 * golden_ratio * i)
            density += np.cos(freq * (kx*self.X + ky*self.Y + kz*self.Z) + self.phi)

        return density


class MaterialCrystal(Crystal):

    def __init__(self, material='silicon', size=1, resolution=30):
        self.materials = {
            'silicon': {
                'lattice_constant': 0.543,
                'structure': 'diamond',
                'wavelength': 0.154
            },
            'graphite': {
                'lattice_constant': 0.246,
                'c_constant': 0.671,
                'structure': 'hexagonal',
                'wavelength': 0.154
            },
            'nacl': {
                'lattice_constant': 0.564,
                'structure': 'fcc',
                'wavelength': 0.154
            },
            'hgse_50nm': {
                'lattice_constant': 0.608,
                'structure': 'zincblende',
                'wavelength': 0.154,
                'thickness': 50
                },
            'hgse_fe_15nm_hgse_50nm': {
                'lattice_constant': 0.608,
                'doped_lattice_constant': 0.610,
                'structure': 'layered_zincblende',
                'wavelength': 0.154,
                'layer1_thickness': 15,
                'layer2_thickness': 50
                }
        }

        self.material = material
        self.params = self.materials[material]
        super().__init__(size=size, resolution=resolution,
                        wavelength=self.params['wavelength'], n=1.0)

    def calculate_structure(self):
        if self.material == 'hgse_50nm':
            return self._zincblende_structure()
        elif self.material == 'hgse_fe_15nm_hgse_50nm':
            return self._layered_zincblende_structure()
        else:
            return super().calculate_structure()

    def _zincblende_structure(self):
        a = self.params['lattice_constant']
        density = np.zeros_like(self.X)

        k1 = 2*np.pi/a * np.array([1, 1, 0])
        k2 = 2*np.pi/a * np.array([1, 0, 1])
        k3 = 2*np.pi/a * np.array([0, 1, 1])

        for k in [k1, k2, k3]:
            density += np.cos(k[0]*self.X + k[1]*self.Y + k[2]*self.Z)

        shift = a/4 * np.array([1, 1, 1])
        for k in [k1, k2, k3]:
            density += np.cos(k[0]*(self.X+shift[0]) +
                       k[1]*(self.Y+shift[1]) +
                       k[2]*(self.Z+shift[2]))

        if 'thickness' in self.params:
            thickness = self.params['thickness']
            z_mask = (self.Z >= (self.size - thickness/self.size))
            density *= z_mask

        return density

    def _layered_zincblende_structure(self):
        a1 = self.params['doped_lattice_constant']
        a2 = self.params['lattice_constant']
        layer1_thick = self.params['layer1_thickness']
        layer2_thick = self.params['layer2_thickness']

        density = np.zeros_like(self.X)

        z_mask1 = (self.Z <= layer1_thick/self.size)
        k1 = 2*np.pi/a1 * np.array([1, 1, 0])
        k2 = 2*np.pi/a1 * np.array([1, 0, 1])
        k3 = 2*np.pi/a1 * np.array([0, 1, 1])

        for k in [k1, k2, k3]:
            density += z_mask1 * np.cos(k[0]*self.X + k[1]*self.Y + k[2]*self.Z)

        z_mask2 = (self.Z > layer1_thick/self.size) & (self.Z <= (layer1_thick+layer2_thick)/self.size)
        k1 = 2*np.pi/a2 * np.array([1, 1, 0])
        k2 = 2*np.pi/a2 * np.array([1, 0, 1])
        k3 = 2*np.pi/a2 * np.array([0, 1, 1])

        for k in [k1, k2, k3]:
            density += z_mask2 * np.cos(k[0]*self.X + k[1]*self.Y + k[2]*self.Z)

        return density


class Visualizer:
    def __init__(self):
        self.crystal = Crystal()
        self.quasicrystal = Quasicrystal()

        self.disk_color = (12/255, 128/255, 128/255)
        self.edge_color = (20/255, 200/255, 200/255)
        self.current_mode = 'crystal'

        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.setup_ui()
        self.update_visualization()

    def setup_ui(self):
        # Радиокнопки для выбора режима
        rax = plt.axes([0.05, 0.7, 0.1, 0.15])
        self.radio = RadioButtons(rax, ('Crystal', 'Quasicrystal'))
        self.radio.on_clicked(self.change_mode)

        # Слайдеры для управления углами обзора
        ax_elev = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_azim = plt.axes([0.25, 0.05, 0.65, 0.03])

        self.slider_elev = Slider(ax_elev, 'Elevation', -180, 180, valinit=30)
        self.slider_azim = Slider(ax_azim, 'Azimuth', -180, 180, valinit=-45)

        # Кнопка сброса
        resetax = plt.axes([0.8, 0.15, 0.1, 0.04])
        self.button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

        # Привязка обработчиков событий
        self.slider_elev.on_changed(self.update_view)
        self.slider_azim.on_changed(self.update_view)
        self.button.on_clicked(self.reset_view)

    def change_mode(self, label):
        self.current_mode = label.lower()
        self.update_visualization()

    def update_view(self, val):
        self.ax.view_init(elev=self.slider_elev.val, azim=self.slider_azim.val)
        self.fig.canvas.draw_idle()

    def reset_view(self, event):
        self.slider_elev.reset()
        self.slider_azim.reset()

    def update_visualization(self, threshold=0.5):
        self.ax.clear()

        try:
            if self.current_mode == 'crystal':
                model = self.crystal
                title = '3D модель кристаллической структуры'
            else:
                model = self.quasicrystal
                title = '3D модель квазикристаллической структуры'

            verts, faces, _, _ = marching_cubes(
                model.density,
                level=threshold*np.max(model.density)
            )

            verts[:, 0] *= model.size / model.resolution
            verts[:, 1] *= model.size / model.resolution
            verts[:, 2] *= model.size / model.resolution

            mesh = Poly3DCollection(verts[faces], alpha=0.8)
            mesh.set_facecolor(self.disk_color)
            mesh.set_edgecolor(self.edge_color)
            self.ax.add_collection3d(mesh)

            self.ax.set_xlim(0, model.size)
            self.ax.set_ylim(0, model.size)
            self.ax.set_zlim(0, model.size)
            self.ax.set_xlabel('X (мкм)')
            self.ax.set_ylabel('Y (мкм)')
            self.ax.set_zlabel('Z (мкм)')
            self.ax.set_title(title)
            self.ax.view_init(elev=30, azim=-45)

        except Exception as e:
            print(f"Ошибка визуализации: {e}")

        plt.tight_layout()

    def show(self):
        plt.show()


class AdvancedVisualizer(Visualizer):
    def __init__(self):
        super().__init__()
        self.crystal = MaterialCrystal('silicon')
        self.setup_material_ui()

    def setup_material_ui(self):
        rax_materials = plt.axes([0.05, 0.4, 0.1, 0.3])
        self.material_radio = RadioButtons(
            rax_materials,
            ('Silicon', 'Graphite', 'NaCl', 'HgSe 50nm', 'HgSe:Fe 15nm\nHgSe 50nm')
        )
        self.material_radio.on_clicked(self.change_material)

    def change_material(self, label):
        material_map = {
            'Silicon': 'silicon',
            'Graphite': 'graphite',
            'NaCl': 'nacl',
            'HgSe 50nm': 'hgse_50nm',
            'HgSe:Fe 15nm\nHgSe 50nm': 'hgse_fe_15nm_hgse_50nm'
        }
        self.crystal = MaterialCrystal(material_map[label])
        self.update_visualization()


if __name__ == "__main__":
    viewer = AdvancedVisualizer()
    viewer.show()
