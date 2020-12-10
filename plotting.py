import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import cm
from math import ceil, sqrt

class BasicPlot(object):
    def __init__(
            self, input, plot_data,
            num_cells_to_plot, fps, select_cells, save_filename
            ):
        self.input = input
        self.num_cells_to_plot = ceil(sqrt(num_cells_to_plot))**2
        self.fps = fps
        self.plot_data = plot_data
        if select_cells is None:
            self.select_cells_to_plot = np.random.choice(
                self.input.num_states, self.num_cells_to_plot
                )
        else:
            self.select_cells_to_plot = select_cells
        self.save_filename = save_filename
        self.init_plot()

    def init_plot(self):
        fig = plt.figure() 
        self.fig = fig
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2])
        num_blocks = sqrt(self.num_cells_to_plot)
        heatmap_size = (self.input.num_xybins+1) * num_blocks
        heatmap_size = int(heatmap_size)
        heatmap = np.zeros((heatmap_size, heatmap_size))*np.nan
        self.ax1 = fig.add_subplot(gs[0,1])
        self.ax2 = fig.add_subplot(gs[1,0])
        self.ax3 = fig.add_subplot(gs[1,1])
        self.ax4 = fig.add_subplot(gs[1,2])
        self.line1, = self.ax1.plot([0], [0], linewidth=0.5)
        self.ax1.set_xlim(self.input.xmin, self.input.xmax)
        self.ax1.set_ylim(self.input.ymin, self.input.ymax)
        self.im2 = self.ax2.imshow(heatmap)
        self.im3 = self.ax3.imshow(heatmap)
        self.im4 = self.ax4.imshow(heatmap)
        titles = ["Trajectory", "Place Cells", "Grid Cells", "Reconstructed Place Cells"]
        axs = [self.ax1, self.ax2, self.ax3, self.ax4]
        for ax, title in zip(axs, titles):
            ax.tick_params(
                axis='both', which='both',
                bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False
                )
            ax.set_title(title)
        plt.tight_layout()

    def init_animation(self):
        num_blocks = sqrt(self.num_cells_to_plot)
        heatmap_size = (self.input.num_xybins+1) * num_blocks
        heatmap_size = int(heatmap_size)
        heatmap = np.zeros((heatmap_size, heatmap_size))*np.nan
        self.im2.set_data(heatmap)
        return self.line1, self.im2, self.im3, self.im4

    def animate(self):
        anim = animation.FuncAnimation(
            self.fig, self.update, init_func=self.init_animation,
            frames=len(self.plot_data), interval=1/self.fps, blit=True
            )
        anim.save(
            self.save_filename, fps=self.fps, extra_args=['-vcodec', 'libx264']
            )
        plt.show()

    def update(self, frame):
        M, U, M_hat, xs, ys = self.plot_data[frame]
        self.line1.set_data(xs, ys)
        self.im2.set_data(self._format_into_heatmap(M))
        self.im2.autoscale()
        self.im3.set_data(self._format_into_heatmap(U, use_selected=False))
        self.im3.autoscale()
        self.im4.set_data(self._format_into_heatmap(M_hat))
        self.im4.autoscale()
        self.im2.cmap.set_bad(color='white')
        self.im3.cmap.set_bad(color='white')
        self.im4.cmap.set_bad(color='white')
        return self.line1, self.im2, self.im3, self.im4

    def _format_into_heatmap(self, mat, limits=[1, 99], use_selected=True):
        num_states, num_cells = mat.shape
        num_blocks = sqrt(self.num_cells_to_plot)
        heatmap_size = (self.input.num_xybins+1) * num_blocks
        heatmap_size = int(heatmap_size)
        heatmap = np.zeros((heatmap_size, heatmap_size))*np.nan
        if use_selected:
            cells_to_plot = self.select_cells_to_plot
        else:
            cells_to_plot= np.arange(self.num_cells_to_plot)
        for idx, cell in enumerate(cells_to_plot):
            vec = mat[:, cell]
            xbins, ybins = self.input.get_xybins(np.arange(vec.size))
            cell_heatmap = np.zeros(
                (self.input.num_xybins, self.input.num_xybins)
                )
            for i, (xbin, ybin) in enumerate(zip(xbins, ybins)):
                cell_heatmap[xbin, ybin] += vec[i]
            start_row = int((idx//num_blocks) * (self.input.num_xybins+1))
            end_row = int(start_row + self.input.num_xybins)
            start_col = int((idx%num_blocks) * (self.input.num_xybins+1))
            end_col = int(start_col + self.input.num_xybins)
            cell_heatmap = cell_heatmap
            heatmap[start_row:end_row, start_col:end_col] = np.flipud(cell_heatmap)
        nonnan_vals = heatmap[np.logical_not(np.isnan(heatmap))].flatten()
        limits = np.percentile(nonnan_vals, [limits[0], limits[1]])
        heatmap = np.clip(heatmap, limits[0], limits[1])
        return heatmap

