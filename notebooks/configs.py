dt_to_sec = 3
bin_to_cm = 5

payne2021_stats = {
    'fieldsize': [0.0752, 0.00697], # [mean, CI]
    'nfields': [1.5986, 0.0993], # [mean, CI]
    'onefield': [0.5917] # [values,...]
    }

henrikson2010_stats = {
    'fieldsize': [0.0725, 0.005], # [mean, CI]
    'nfields': [1.6, 0.167], # [mean, CI]
    'onefield': [0.57, 0.532] # [values,...]
    }

import seaborn as sns
sns.set(font='Arial',
        font_scale=16/12., #default size is 12pt, scale to 16pt
        palette='Set1',
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
