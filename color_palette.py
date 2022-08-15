'''
This stores custom color palettes.

Overview on choice of colors:
A practical guide on (HSLuv): https://seaborn.pydata.org/tutorial/color_palettes.html

'''
import matplotlib as mpl
import seaborn as sns
import numpy as np

## Color palette tools
## Coolors: https://coolors.co/

# QUANTITATIVE
## N = 5
## For a new color palette, try
#### 1. Colormind: http://colormind.io/
#### 2. Color-hex: https://www.color-hex.com/color-palettes/popular.php
porcoRosso = ["1f448c","ba3625","be944d","567a36","592524"]

# CUSTOM (HELICAL TURB)
myPalette5 = np.asarray([
                     [0.2298057 , 0.29871797, 0.75368315, 1.        ], # Left-handed: blue, #3B4CC0
                     [0.18431373, 0.49019608, 0.33333333, 1.        ], # Non-helical 1:, #4A7D55, green
                     [0.85490196, 0.64705882, 0.12549020, 1.        ], # Non-helical 2:, #DAA520, Gold
                     [0.70567316, 0.01555616, 0.15023281, 1.        ], # Right-handed: maroon, #C80426
                     [0.2, 0.2, 0.2, 1.        ], # Planar (black), #333333
                    ])

## N = 3
myPalette3 = np.asarray([
                     [0.2298057 , 0.29871797, 0.75368315, 1.        ], # Left-handed: blue, #3B4CC0
                     [0.85490196, 0.64705882, 0.12549020, 1.        ], # Non-helical 2, #DAA520, Gold
                     [0.70567316, 0.01555616, 0.15023281, 1.        ], # Right-handed: maroon, #C80426
                    ])

## N = 7-10 (Color-blind friendly)
blind8_okabe = ['56B4E9', 'E69F00', '000000', 'CC79A7', 'D55E00', '0072B2', 'F0E442', '009E73']
blind7_tol_bright = ['228833', 'EE6677', 'BBBBBB', 'AA3377', '66CCEE', 'CCBB44', '4477AA']
blind9_tol_light = ['AAAA00', 'BBCC33', 'DDDDDD', '44BB99', '99DDFF',
                     'FFAABB', 'EEDD88', 'EE8866', '77AADD']
blind10_tol_muted = ['117733', '44AA99', '88CCEE', 'DDDDDD', 'AA4499',
                     '882255', 'CC6677', '999933', 'DDCC77', '332288']



## SEQUENTIAL
skypiea= ['001219', '005f73', '0a9396', '94d2bd', 'e9d8a6', 'ee9b00', 'ca6702', 'bb3e03', 'ae2012', '9b2226']
sunset = ['03071e', '370617', '6a040f', '9d0208', 'd00000', 'dc2f02', 'e85d04', 'f48c06', 'faa307', 'ffba08']


def hex2rgb(hex, normalize=False):
    """
    Converts a HEX code to RGB in a numpy array
    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    h = hex.strip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) / 255. for i in (0, 2, 4)))
    return rgb

def generate_cmap_from_colors(colors):
    """
    Generates a colormap from a given list of colors. It accepts oth RGBA values and Hex codes.

    Parameters
    ----------
    colors: a list of rgba values (range: 0-1) or hex codes

    Returns
    -------
    cmap: color map, matplotlib.colors.LinearSegmentedColormap object
    """

    if type(colors[0]) == str:
        colors_rgb = list(map(hex2rgb, colors))
    else:
        colors_rgb = colors
    cmap = mpl.colors.ListedColormap(colors_rgb)
    return cmap

