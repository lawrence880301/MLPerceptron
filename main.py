from math import e
from math import ceil, floor
from DataPreprocessor import Datapreprocessor
from SingleLayerPerceptron import SingleLayerPerceptron
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import random
from operator import itemgetter
from test import Perceptron

##########code###########
def openfile():
    global dataset_url, filename
    filename.set("")
    dataset_url = filedialog.askopenfilename(title="Select file")
    filename.set(dataset_url.split("/")[-1].split(".")[0])
    return dataset_url

def prepare_data():
    dataset = Datapreprocessor.readfile(dataset_url)
    dataset = Datapreprocessor.text_to_numlist(dataset)
    return dataset

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def start():
    #refresh previous weights and score
    trained_weights.set("")
    trained_score.set("")
    #prepare dataset and train(data should be modified to 0 and 1)
    dataset= prepare_data()
    model = Perceptron(len(dataset[1])-1, int(entry_iteration.get()), float(entry_learning_rate.get()))
    label_modified_dataset = Datapreprocessor.label_preprocess(dataset)
    modified_train_data, modified_test_data = Datapreprocessor.train_test_split(label_modified_dataset, 2/3)
    modified_train_x, modified_train_y = Datapreprocessor.feature_label_split(modified_train_data)
    modified_test_x, modified_test_y = Datapreprocessor.feature_label_split(modified_test_data)
    model.train(np.array(modified_train_x), np.array(modified_train_y))
    #predict and setting current weights and score
    trained_weights.set(model.weights)
    predict_list = []
    for row in modified_test_x:
        predict = model.predict(row)
        predict_list.append(predict)
    test_accuaracy = accuracy_metric(modified_test_y ,predict_list)
    trained_score.set(test_accuaracy)
    #plot line and data point
    plot(model.weights, dataset, modified_train_x, modified_train_y, modified_test_x, modified_test_y)

def plot(weights, dataset, modified_train_x, modified_train_y, modified_test_x, modified_test_y):
    fig.clear()
    origin_train_data, origin_test_data = Datapreprocessor.train_test_split(dataset, 2/3)
    origin_train_x, origin_train_y = Datapreprocessor.feature_label_split(origin_train_data)
    origin_test_x, origin_test_y = Datapreprocessor.feature_label_split(origin_test_data)
    # 2d
    print(modified_train_x[0])
    if len(modified_train_x[0]) == 2:
        plot1 = fig.add_subplot(111)
        min_x = floor(min([row[0] for row in modified_train_x]))
        max_x = ceil(max([row[0] for row in modified_train_x]))
        origin_min_x = floor(min([row[0] for row in origin_train_data]))
        origin_max_x = ceil(max([row[0] for row in origin_train_data]))
        origin_min_y = floor(min([row[1] for row in origin_train_data]))
        origin_max_y = ceil(max([row[1] for row in origin_train_data])) 
        x = np.array(range(min_x, max_x+1))
        origin_x = np.array(range(origin_min_x, origin_max_x))
        origin_y = np.array(range(origin_min_y, origin_max_y))
        # if weights[1]!=0:
        y = (-1*(weights[1]+ 1*e-20)*x + (weights[0]+1*e-20)) / (weights[2]+1*e-20)
        # plot data point
        plot1.set_xticks(origin_x)
        plot1.set_yticks(origin_y)
        plot1.plot(x,y)
    canvas.draw()

    #     train_set = Datapreprocessor.label_preprocess(train_set)
    #     test_set = Datapreprocessor.label_preprocess(test_set)
    #     train_set.sort(key=itemgetter(-1), reverse=False)
    #     test_set.sort(key=itemgetter(-1), reverse=False)
    #     train_data = pd.DataFrame({"X Value": [row[0] for row in train_set], "Y Value": [row[1] for row in train_set], "Category": [row[2] for row in train_set]})
    #     test_data = pd.DataFrame({"X Value": [row[0] for row in test_set], "Y Value": [row[1] for row in test_set], "Category": [row[2] for row in test_set]})
    #     train_groups = train_data.groupby("Category")
    #     test_groups = test_data.groupby("Category")
        
    #     group_idx_list = []
    #     for group_idx, _ in train_groups:
    #         if group_idx not in group_idx_list:
    #             group_idx_list.append(group_idx)
    #     color_dict = {}
    #     color_dict[group_idx_list[0]] = "r"
    #     color_dict[group_idx_list[1]] = "b"


    #     for name, group in train_groups:
    #         plot1.plot(group["X Value"], group["Y Value"], marker="o", c = color_dict[name],linestyle="", label=name)
    #     for name, group in test_groups:
    #         plot1.plot(group["X Value"], group["Y Value"], marker="x", c = color_dict[name],linestyle="", label=name)
    # if len(train_set[0])-1 == 3:
    #     plot2 = fig.add_subplot(1, 2, 1, projection='3d')
    #     min_x = floor(min([row[0] for row in train_set]))
    #     max_x = ceil(max([row[0] for row in train_set]))
    #     x = np.array(range(min_x, max_x+1))
    #     min_y = floor(min([row[1] for row in train_set]))
    #     max_y = ceil(max([row[1] for row in train_set]))
    #     y = np.array(range(min_y, max_y+1))
    #     X, Y = np.meshgrid(x, y)
    #     z = (-1*(weights[1]+ 1*e-20)*X + -1*(weights[2]+ 1*e-20)*Y +  (weights[0]+1*e-20)) / (weights[3]+1*e-20)
    #     plot2.plot_surface(x, y, z)
        
    #     train_set.sort(key=itemgetter(-1), reverse=False)
    #     test_set.sort(key=itemgetter(-1), reverse=False)
    #     train_data = pd.DataFrame({"X Value": [row[0] for row in train_set], "Y Value": [row[1] for row in train_set], "Z Value": [row[2] for row in train_set],"Category": [row[3] for row in train_set]})
    #     test_data = pd.DataFrame({"X Value": [row[0] for row in test_set], "Y Value": [row[1] for row in test_set], "Z Value": [row[2] for row in test_set], "Category": [row[3] for row in test_set]})
    #     train_groups = train_data.groupby("Category")
    #     test_groups = test_data.groupby("Category")
        
    #     group_idx_list = []
    #     for group_idx, _ in train_groups:
    #         if group_idx not in group_idx_list:
    #             group_idx_list.append(group_idx)
    #     color_dict = {}
    #     color_dict[group_idx_list[0]] = "r"
    #     color_dict[group_idx_list[1]] = "b"


    #     for name, group in train_groups:
    #         plot2.plot(group["X Value"], group["Y Value"],group["Z Value"], marker="o", c = color_dict[name],linestyle="", label=name)
    #     for name, group in test_groups:
    #         plot2.plot(group["X Value"], group["Y Value"], group["Z Value"], marker="x", c = color_dict[name],linestyle="", label=name)
        





############gui##############
window = tk.Tk()
window.title('perceptron')
window.geometry('600x600')

# model parameter
learning_rate = tk.DoubleVar()
entry_learning_rate = tk.Entry(window, textvariable=learning_rate)
entry_learning_rate.place(x=160, y=100)
tk.Label(window, text='learning rate: ').place(x=50, y= 100)
iteration = tk.IntVar()
entry_iteration = tk.Entry(window, textvariable=iteration)
entry_iteration.place(x=160, y=140)
tk.Label(window, text='iteration: ').place(x=50, y=140)

#select data
dataset_url = ""
filename = tk.StringVar()
selected_file = tk.Label(window, textvariable=filename).place(x=280, y= 50)
select_data_btn = tk.Button(window, text='select dataset', command=openfile).place(x=160, y=50)

#start
btn_start = tk.Button(window, text='start', command=start)
btn_start.place(x=50, y=180)

#weights
tk.Label(window, text='trained_weights: ').place(x=50, y= 230)
trained_weights = tk.StringVar()
lb_trained_weights = tk.Label(window, textvariable=trained_weights).place(x=160, y=230)

#score
tk.Label(window, text='score: ').place(x=50, y= 270)
trained_score = tk.StringVar()
lb_trained_score = tk.Label(window, textvariable=trained_score).place(x=160, y=270)

tk.Label(window, text='score: ').place(x=50, y= 310)
test_score = tk.StringVar()
lb_trained_score = tk.Label(window, textvariable=test_score).place(x=160, y=310)

#plot
# the figure that will contain the plot
fig = Figure(figsize = (5, 5),
        dpi = 100)

# adding the subplot

# creating the Tkinter canvas
# containing the Matplotlib figure
canvas = FigureCanvasTkAgg(fig,
                            master = window) 



# placing the canvas on the Tkinter window
canvas.get_tk_widget().pack()

# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas,
                                window)
toolbar.update()

# placing the toolbar on the Tkinter window
canvas.get_tk_widget().pack()

window.mainloop()









