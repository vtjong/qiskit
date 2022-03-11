from matplotlib import rcParams
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import gzip
import pickle as pk
######################## FONT CONFIGURATION ############################
# plt.rcParams["font.family"] = "AppleGothic"
########################################################################
rcParams["axes.linewidth"] = 0.15
today = date.today()
today = today.strftime("%b-%d-%Y")

def training_plot(fig, qnn_data_path="trainHistoryDict", cnn_data_path="Mar-10-2022ccnn_loss.csv"): 
    """
    For plotting raw input and output data after preprocessing (normalization, etc). 
    [num] is the number of samples we want to plot; plotting more than 5 is 
    difficult to read
    npy version
    """
    with open(qnn_data_path, "rb") as f2:
        qnn_data = pk.load(f2)
    
    plt.plot(qnn_data['loss'], label='QHCCNN Training Loss', linewidth=1)

    cnn_data = np.loadtxt(cnn_data_path).flatten()
    plt.plot(cnn_data, label='CCNN Training Loss', linewidth=1)

    plotname= "Training Loss"
    x_name= "Epochs"
    y_name= "Cross Entropy Loss"

    return plotname, x_name, y_name

def test_plot(fig, qnn_data_path="trainHistoryDict", cnn_data_path="Mar-10-2022test_loss.csv"): 
    """
    For plotting raw input and output data after preprocessing (normalization, etc). 
    [num] is the number of samples we want to plot; plotting more than 5 is 
    difficult to read
    npy version
    """
    with open(qnn_data_path, "rb") as f2:
        qnn_data = pk.load(f2)
    
    plt.plot(qnn_data['val_loss'][::5], label='QHCCNN Validation Loss', linewidth=1)

    cnn_data = np.loadtxt(cnn_data_path).flatten()
    plt.plot(cnn_data, label='CCNN Validation Loss', linewidth=1)

    plotname= "Validation Loss"
    x_name= "Epochs"
    y_name= "Cross Entropy Loss"

    return plotname, x_name, y_name

def adjustFigAspect(fig, aspect=1):
    """
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
 
    aspect: float value that regulates x to y ratio
    """
    xsize, ysize = fig.get_size_inches()
    minsize = min(xsize, ysize)
    xlim = 0.4 * minsize / xsize
    ylim = 0.4 * minsize / ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(
        left=0.5 - xlim, right=0.5 + xlim, bottom=0.5 - ylim, top=0.5 + ylim
    )

# Make the plot
fig = plt.figure()
adjustFigAspect(fig, aspect=1.7)

# plotname, x_name, y_name = training_plot(fig)
plotname, x_name, y_name = test_plot(fig)
plt.title(plotname)
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.legend(loc="upper right")
plt.show()