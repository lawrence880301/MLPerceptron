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



#gui
window = tk.Tk()
window.title('perceptron')
window.geometry('600x600')

# welcome image
canvas = tk.Canvas(window, height=400, width=400)
# image_file = tk.PhotoImage(file='welcome.gif')
# image = canvas.create_image(0,0, anchor='nw', image=image_file)
canvas.pack(side='top')

# model parameter
learning_rate = tk.DoubleVar()
entry_learning_rate = tk.Entry(window, textvariable=learning_rate)
entry_learning_rate.place(x=160, y=230)
tk.Label(window, text='learning rate: ').place(x=50, y= 230)
iteration = tk.IntVar()
entry_iteration = tk.Entry(window, textvariable=iteration)
entry_iteration.place(x=160, y=270)
tk.Label(window, text='iteration: ').place(x=50, y= 270)


def start():
    trained_weights.set("")
    trained_score.set("")
    dataset= prepare_data()
    model = SingleLayerPerceptron(learning_rate.get(), iteration.get())
    scores = model.evaluate_algorithm(dataset, 'perceptron')
    trained_weights.set(model.weights)
    trained_score.set(scores)
    plot(model.weights, model.train_set, model.test_set)



#select data
dataset_url = ""
filename = tk.StringVar()
selected_file = tk.Label(window, textvariable=filename).place(x=280, y= 110)
select_data_btn = tk.Button(window, text='select dataset', command=openfile).place(x=160, y=110)

#start
btn_login = tk.Button(window, text='start', command=start)
btn_login.place(x=160, y=310)

#weights
tk.Label(window, text='trained_weights: ').place(x=50, y= 330)
trained_weights = tk.StringVar()
lb_trained_weights = tk.Label(window, textvariable=trained_weights).place(x=160, y=330)

#score
tk.Label(window, text='score: ').place(x=50, y= 370)
trained_score = tk.StringVar()
lb_trained_score = tk.Label(window, textvariable=trained_score).place(x=160, y=370)

#plot
# the figure that will contain the plot
fig = Figure(figsize = (5, 5),
        dpi = 100)

# adding the subplot

# creating the Tkinter canvas
# containing the Matplotlib figure
canvas = FigureCanvasTkAgg(fig,
                            master = window)  

def plot(weights, train_set, test_set):
    fig.clear()

    # 2d
    if len(train_set[0])-1 == 2:
        plot1 = fig.add_subplot(111)
        min_x = floor(min([row[0] for row in train_set]))
        max_x = ceil(max([row[0] for row in train_set]))
        x = np.array(range(min_x, max_x+1))
        # if weights[1]!=0:
        y = (-1*(weights[1]+ 1*e-20)*x + (weights[0]+1*e-20)) / (weights[2]+1*e-20)

        # plot data point
        plot1.plot(x,y)
        train_data = pd.DataFrame({"X Value": [row[0] for row in train_set], "Y Value": [row[1] for row in train_set], "Category": [row[2] for row in train_set]})
        test_data = pd.DataFrame({"X Value": [row[0] for row in test_set], "Y Value": [row[1] for row in test_set], "Category": [row[2] for row in test_set]})
        train_groups = train_data.groupby("Category")
        test_groups = test_data.groupby("Category")
        
        group_idx_list = []
        for group_idx, _ in train_groups:
            if group_idx not in group_idx_list:
                group_idx_list.append(group_idx)
        color_dict = {}
        color_dict[group_idx_list[0]] = "r"
        color_dict[group_idx_list[1]] = "b"


        for name, group in train_groups:
            plot1.plot(group["X Value"], group["Y Value"], marker="o", c = color_dict[name],linestyle="", label=name)
        for name, group in test_groups:
            plot1.plot(group["X Value"], group["Y Value"], marker="x", c = color_dict[name],linestyle="", label=name)
    if len(train_set[0])-1 == 3:
        plot2 = fig.add_subplot(1, 2, 1, projection='3d')
        z_line = np.linspace(0, 15, 1000)
        x_line = np.cos(z_line)
        y_line = np.sin(z_line)
        plot2.plot3D(x_line, y_line, z_line, 'gray')
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()

def clear():
    for item in canvas.get_tk_widget().find_all():
       canvas.get_tk_widget().delete(item)

window.mainloop()









