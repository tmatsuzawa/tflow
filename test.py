import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np


fig = plt.figure(figsize=(12,12))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))


x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

# sc = ax1.scatter(x, y, c=c, s=100, cmap=cmap, norm=norm)

annot = ax1.annotate("", xy=(0, 0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def show_param_onclick(event):
    button = ['', 'left', 'middle', 'right']
    if event.inaxes is not None:
        ax = event.inaxes
        if ax in [ax1]:
            ax_ind = [ax1].index(ax)
            x, y = event.xdata, event.ydata
            annot.xy = [x, y]
            if button[event.button]=='left':
                text = 'test'
                annot.set_text(text)
                annot.set_visible(True)
                fig.canvas.draw()
            if button[event.button] == 'right':
                annot.set_visible(False)
                fig.canvas.draw()


fig.canvas.mpl_connect("button_press_event", show_param_onclick)


plt.show()