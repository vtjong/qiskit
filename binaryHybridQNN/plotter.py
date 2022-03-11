from matplotlib import rcParams
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
######################## FONT CONFIGURATION ############################
# plt.rcParams["font.family"] = "AppleGothic"
########################################################################
rcParams["axes.linewidth"] = 0.15
today = date.today()
today = today.strftime("%b-%d-%Y")

def raw_data_plot(fig, num, data_path="devices/Oct 07 1000 trials outputs.npy"): 
    """
    For plotting raw input and output data after preprocessing (normalization, etc). 
    [num] is the number of samples we want to plot; plotting more than 5 is 
    difficult to read
    npy version
    """
    data = np.loadtxt(data_path).flatten()
    plt.plot(data, linewidth=1)

def result_plot(actual_path, predicted_path, fig, diff=False, numpoints=100):
    """
    For plotting mode value of [actual_path] versus [predicted_path].
    Set [diff] = True for a plot of predicted - actual testing data.
    """
    predict = np.loadtxt(predicted_path).flatten()[:numpoints]
    actual = np.loadtxt(actual_path).flatten()[:numpoints]
    max_count = predict.size
    count = list(range(1, max_count + 1))
    if diff:
        plt.plot(
            count,
            predict - actual,
            color="#50514F",
            label="predict - actual",
            linewidth=3,
        )
    else:
        plt.plot(count, predict, label="predict", linewidth=2)
        plt.plot(count, actual, label="actual", linewidth=1)
        # plt.plot(count, predict, color='#ea75a3',  label='predict', linewidth=4)
        # plt.plot(count, actual, color='#50514F',  label='actual', linewidth=3)

    plotname = "100 Simulation Actual v. Predicted"
    x_name = "Count"
    y_name = "Value"
    return max_count, plotname, x_name, y_name

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
# raw_data_plot(
#     fig, 1, data_path="/Users/valenetjong/qiskit/Feb-23-2022hybrid_loss.csv")
# plt.title("Hybrid Net Loss")
# plt.xlabel('Epochs')
# plt.ylabel('Neg Log Likelihood Loss')

result_plot(actual_path='Feb-23-2022vers2_actuals_nonlinear.csv', predicted_path='Feb-23-2022vers2_actuals_nonlinear.csv',fig=fig)
plt.title("Actuals v. Predicted")
plt.xlabel("Count")
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.show()