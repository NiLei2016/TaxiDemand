import numpy as np
import pandas as pd
import time ,datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import matplotlib
from matplotlib.font_manager import FontProperties
zhfont1 = FontProperties(fname='C:\Windows\Fonts\simkai.ttf',size=14)
def read_chunk(dir):  # data block reading
    reader = pd.read_csv(dir, iterator=True)
    loop = True
    chunk_size = 1000000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=False)
    return df
def plot_broken_line(): # plot the different lenght of closeness
    dir = "H:\\NewYork Taxi Data\\ConvLSTM_len_period.txt"
    f = open(dir, 'r')
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    line = f.readline()
    while line:
        x, y1, y2, y3, y4 = line.split(",")
        X.append(int(x))
        Y1.append(float(y1))
        Y2.append(float(y2))
        Y3.append(float(y3))
        Y4.append(float(y4))
        line = f.readline()
    f.close()
    print(X, Y1, Y2, Y3, Y4)
    fig = plt.figure()
    plt.axis([1, 5, 1, 40])
    ticks = [0, 1, 2, 3, 4, 5]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(X, Y1, 'r--', label=u'5 min')
    plt.plot(X, Y2, 'g--', label=u'15 min')
    plt.plot(X, Y3, 'b--', label=u'30 min')
    plt.plot(X, Y4, 'y--', label=u'60 min')
    plt.xlabel('length of period lp', fontsize=10)
    plt.ylabel('rmse', fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.xticks(range(0, 6))
    fig.savefig("H:\\NewYork Taxi Data\\Figure of result\\length_of_period.eps")
    fig.savefig("H:\\NewYork Taxi Data\\Figure of result\\length_of_period.png")
    plt.show()
def plot_prediction_Truth(no):
    dir1 = "H:\\NewYork Taxi Data\\prediction 2"
    dir2 = 'H:\\NewYork Taxi Data\\Prediciton CNNLSTM'
    Y1 = []
    Y2 = []
    Y3 = []
    f = open(os.path.join(dir1, "30min_ground truth.txt"))
    line = f.readline()
    while line:
        y1 = line.split(" ")[no]
        Y1.append(float(y1))
        line = f.readline()
    f.close()
    f = open(os.path.join(dir2, "30min_prediction.txt"))
    line = f.readline()
    while line:
        y2 = line.split(" ")[no]
        Y2.append(float(y2))
        line = f.readline()
    f.close()
    f = open(os.path.join(dir1, "30min_prediction.txt"))
    line = f.readline()
    while line:
        y3 = line.split(" ")[no]
        Y3.append(float(y3))
        line = f.readline()
    f.close()
    lables = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ticks = [12, 36, 60, 84, 108, 132, 156]
    fig = plt.figure(figsize=(10, 6))
    #plt.axis([1, 168, 1500, 20000])
    plt.plot(range(1, 169), Y1, 'g--', label='Ground Truth')
    plt.plot(range(1, 169), Y2, 'r-', label='CNN+LSTM')
    plt.plot(range(1, 169), Y3, 'b-', label='ConvLSTM')
    plt.xlabel('Time')
    plt.ylabel('Number of requests')
    plt.xticks(ticks, lables)
    plt.title('Taxi demand')
    plt.legend(loc='upper left', fontsize=10)
    fig.savefig("H:\\NewYork Taxi Data\\Figure of Result 4\\30 min\\30min_prediction_truth_"+str(no)+".png")
    fig.savefig("H:\\NewYork Taxi Data\\Figure of Result 4\\30 min\\30min_prediction_truth_"+str(no)+".eps")
    #plt.show()


def plot_each_area():
    for i in range(1, 101):
        plot_prediction_Truth(i)

def plot_hist():
    dir = "H:\\NewYork Taxi Data\\Result of differ method.txt"
    f = open(dir, 'r')
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    line = f.readline()
    while line:
        print(line)
        x, y1, y2, y3, y4 = line.split(",")
        X.append(int(x))
        Y1.append(float(y1))
        Y2.append(float(y2))
        Y3.append(float(y3))
        Y4.append(float(y4))
        line = f.readline()
    f.close()
    print(X, Y1, Y2, Y3, Y4)
    total_width, n = 0.8, 4
    width = total_width/n

    X = [x-3/2*width for x in X]
    fig = plt.figure(figsize=(15, 15))
    lables = ['ConvLSTM-MTL', 'ConvLSTM-MTL-noExt', 'CNN+LSTM', 'LSTM', 'GRU', 'STResNet', 'DeepST']
    plt.bar(X, Y1, width=width, alpha=0.5, label='5 min')
    #plt.plot(X, Y1, 'b--')
    X = [x + width for x in X]
    plt.bar(X, Y2, width=width, label='15 min')
    X = [x + width for x in X]
    plt.bar(X, Y3, width=width, label='30 min')
    X = [x + width for x in X]
    plt.bar(X, Y4, width=width, label='60 min')

    plt.ylabel('rmse')
    #plt.xlabel('STL vs MTL')
    plt.xticks(range(1, 8), lables, rotation=90)
    plt.legend(loc='upper left', prop={'size': 12})
    fig.savefig("H:\\NewYork Taxi Data\\Figure of result\\rank differ method.png")
    fig.savefig("H:\\NewYork Taxi Data\\Figure of result\\rank differ method.eps")
    plt.show()
def plot_order_one_week():
    dir = "H:\\NewYork Taxi Data\\First 7 days of June .txt"
    f = open(dir, 'r')
    X = []
    Y = []
    line = f.readline()
    while line:
        x, y1 = line.split(",")
        X.append(int(x))
        Y.append(int(y1))
        line = f.readline()
    f.close()
    lables = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ticks = [12, 36, 60, 84, 108, 132, 156]
    fig = plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'k--')
    plt.xlabel('Time')
    plt.ylabel('Number of requests')
    plt.xticks(ticks, lables)
    plt.title('Taxi demand')
    fig.savefig("H:\\NewYork Taxi Data\\Figure of result\\Yellow Taxi demand for one week.png")
    plt.show()
def plot_order_one_day():
    dir = "H:\\NewYork Taxi Data\\24 hours order rain.txt"
    f = open(dir, 'r')
    X = []
    Y = []
    line = f.readline()
    while line:
        x, y1 = line.split(",")
        X.append(int(x))
        Y.append(int(y1))
        line = f.readline()
    f.close()
    lables = ['01:00:00', '04:00:00', '07:00:00', '10:00:00', '13:00:00', '16:00:00', '19:00:00','22:00:00']
    ticks = [2, 5, 8, 11, 14, 17, 20, 23]
    fig = plt.figure(figsize=(8, 6))
    plt.plot(X, Y, 'k--', marker='.')
    plt.xlabel(u'时间', fontproperties= zhfont1)
    plt.ylabel('Number of requests')
    plt.xticks(ticks, lables)
    plt.title('Taxi demand')
    plt.annotate('Heavy Rain', xy=(15, 11309),
                 xytext=(14, 14000),
                 arrowprops=dict(arrowstyle='->')),
    xy1 = (13.9, 10200)
    currentAxis = plt.gca()
    rectangle = patches.Rectangle(xy1, 2, 1500, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rectangle)
    #plt.grid(True)
    #fig.savefig("H:\\NewYork Taxi Data\\Figure of result\\Yellow Taxi demand for one day rain.png")
    plt.show()


if __name__ == '__main__':
    #plot_order_one_day()
    plot_hist()