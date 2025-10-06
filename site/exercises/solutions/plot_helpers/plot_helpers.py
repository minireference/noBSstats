"""
This file contains helper functions for plotting the probability distributions.
(c) 2024 Minireference Co. - MIT License

TODOs:
 - change x to xs (to signal it's a array-like)
 - rename all r.v. generation functions to use `gen_` prefix.
"""

# Global figures settings (reused in all notebooks)
RCPARAMS = {
    'figure.figsize': (7,4),     # overriden on base-by-case basis
    'font.serif': ['Palatino',   # as per Minireference Co. style guide
                   # backup fonts in case the Palatino is not available
                   'DejaVu Serif', 'Bitstream Vera Serif',
                   'Computer Modern Roman', 'New Century Schoolbook',
                   'Century Schoolbook L', 'Utopia', 'ITC Bookman',
                   'Bookman', 'Nimbus Roman No9 L', 'Times New Roman',
                   'Times', 'Charter', 'serif'],
    'font.family': 'serif',
    # 'text.latex.preamble': r'\usepackage{amsmath}',  # use for more commands
    #     'figure.dpi': 300,
    #     'font.size': 20,
    #     'figure.titlesize': 16,
    #     'axes.titlesize':22,
    #     'axes.labelsize':20,
    #     'xtick.labelsize': 12,
    #     'ytick.labelsize': 12,
    #     'legend.fontsize': 16,
    #     'legend.title_fontsize': 18,
}

# EACH NOTEBOOK SHOULD INCLUDE SOMETHING LIKE

# # Figures setup
# DESTDIR = "figures/stats/confidence_intervals"
# plt.clf()  # needed otherwise `sns.set_theme` doesn't work
# from plot_helpers import RCPARAMS
# RCPARAMS.update({'figure.figsize': (10, 3)})   # good for screen
# # RCPARAMS.update({'figure.figsize': (5, 1.6)})  # good for print
# sns.set_theme(
#     context="paper",
#     style="whitegrid",
#     palette="colorblind",
#     rc=RCPARAMS,
# )
# # High-resolution please
# %config InlineBackend.figure_format = 'retina'

# Not used:
#    sns.set(color_codes=True)              # turn on Seaborn styles
#    plt.rc('text', usetex=True)            # enable latex for labels
#    ALT palette="colorblind"
#    ALT sns.color_palette('Blues', 4)


# The plot helper functions that used to be in this file have been moved to
# https://github.com/minireference/ministats/blob/main/ministats/plots.py
# and https://github.com/minireference/ministats/blob/main/ministats/utils.py
