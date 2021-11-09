from dataclasses import dataclass
import numpy as np

dt_to_sec = 3
bin_to_cm = 5
fig_width = 6.4
fig_height = 4.8

@dataclass
class Stats:
    mean: float
    ci: float
    std: float
    num_samples: int

@dataclass
class DatasetStats:
    fieldsize: Stats
    nfield: Stats
    onefield: Stats
    fieldsize_distribution: np.ndarray # (shape, scale) of gamma distribution
    nfield_distribution: np.ndarray # PMF of categorical distribution

payne2021 = DatasetStats(
    Stats(0.0752, 0.00697, 0.0764, 462),
    Stats(1.5986, 0.0993, 0.8609, 289),
    Stats(0.5917, None, 0.0302, None),
    np.array([1.12048, 0.067097])
    np.array([0.5917, 0.2664, 0.1003, 0.0381, 0.0035]) # [1, 2, 3, 4, 5+]
    )

henrikson2010 = DatasetStats(
    Stats(0.0725, 0.005, None, None),
    Stats(1.6, 0.167, None, None),
    Stats(np.mean([0.57, 0.532]), None, None, None),
    None
    None
    )

import seaborn as sns
sns.set(font='Arial',
        font_scale=16/12., #default size is 12pt, scale to 16pt
        palette='colorblind', #'Set1',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'text.color': 'dimgrey', #e.g. legend

            'lines.solid_capstyle': 'round',
            'legend.facecolor': 'white',
            'legend.framealpha':0.8,

            'xtick.bottom': True,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',

            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': True,

             'xtick.major.size': 2,
             'xtick.major.width': .5,
             'xtick.minor.size': 1,
             'xtick.minor.width': .5,

             'ytick.major.size': 2,
             'ytick.major.width': .5,
             'ytick.minor.size': 1,
             'ytick.minor.width': .5})
