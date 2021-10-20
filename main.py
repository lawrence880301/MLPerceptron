from math import e
from math import ceil, floor
from DataPreprocessor import Datapreprocessor
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
from Perceptron import Perceptron

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

def predict_list(model, feature):
    predictions = []
    for row in feature:
        predict = model.predict(row)
        predictions.append(predict)   
    return predictions

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
    predict_train_list = predict_list(model, modified_train_x)
    predict_test_list = predict_list(model, modified_test_x)
    train_accuaracy = accuracy_metric(modified_train_y ,predict_train_list)
    test_accuaracy = accuracy_metric(modified_test_y ,predict_test_list)
    trained_score.set(train_accuaracy)
    test_score.set(test_accuaracy)
    #plot line and data point
    plot(model.weights,  modified_train_data, modified_test_data)

def plot(weights,  modified_train_data, modified_test_data):
    fig.clear()
    train_label_0, train_label_1 = Datapreprocessor.group_dataset_by_label(modified_train_data)
    test_label_0, test_label_1 = Datapreprocessor.group_dataset_by_label(modified_test_data)

    min_x = floor(min([row[0] for row in modified_train_data]))
    max_x = ceil(max([row[0] for row in modified_train_data]))
    x = np.array(range(min_x, max_x+1))
    # 2d
    if len(modified_train_data[0]) - 1 == 2:
        plot2d = fig.add_subplot(111)
        y = (-1*(weights[1])*x - weights[0]) / (weights[2])
        plot2d.plot(x,y)
        plot2d.scatter(np.array(train_label_0)[:,0], np.array(train_label_0)[:,1], c="red", marker = 'o')
        plot2d.scatter(np.array(train_label_1)[:,0], np.array(train_label_1)[:,1], c="blue", marker = 'o')
        if len(test_label_0)==1:
            plot2d.scatter(np.array(test_label_0)[0][0], np.array(test_label_0)[0][1], c="red", marker = 'x')
        elif len(test_label_0)>1:
            plot2d.scatter(np.array(test_label_0)[:,0], np.array(test_label_0)[:,1], c="red", marker = 'x')
        if len(test_label_1)==1:
            plot2d.scatter(np.array(test_label_1)[0][0], np.array(test_label_1)[0][1], c="blue", marker = 'x')
        elif len(test_label_1)>1:
            plot2d.scatter(np.array(test_label_1)[:,0], np.array(test_label_1)[:,1], c="blue", marker = 'x')
    #3d
    if len(modified_train_data[0])-1 == 3:
        plot3d = fig.add_subplot(1, 2, 1, projection='3d')
        min_y = floor(min([row[1] for row in modified_train_data]))
        max_y = ceil(max([row[1] for row in modified_train_data]))
        y = np.array(range(min_y, max_y+1))
        X, Y = np.meshgrid(x, y)
        z = (-1*(weights[1])*X + -1*(weights[2])*Y -  (weights[0])) / (weights[3])
        
        plot3d.scatter(np.array(train_label_0)[:,0], np.array(train_label_0)[:,1], np.array(train_label_0)[:,2], c="red", marker = 'o')
        plot3d.scatter(np.array(train_label_1)[:,0], np.array(train_label_1)[:,1], np.array(train_label_1)[:,2], c="blue", marker = 'o')
        if len(test_label_0)==1:
            plot3d.scatter(np.array(test_label_0)[0][0], np.array(test_label_0)[0][1], np.array(test_label_0)[0][2], c="red", marker = 'x')
        elif len(test_label_0)>1:
            plot3d.scatter(np.array(test_label_0)[:,0], np.array(test_label_0)[:,1], np.array(test_label_0)[:,2], c="red", marker = 'x')
        if len(test_label_1)==1:
            plot3d.scatter(np.array(test_label_1)[0][0], np.array(test_label_1)[0][1], np.array(test_label_1)[0][2], c="blue", marker = 'x')
        elif len(test_label_1)>1:
            plot3d.scatter(np.array(test_label_1)[:,0], np.array(test_label_1)[:,1], np.array(test_label_1)[:,2], c="blue", marker = 'x')
        plot3d.plot_surface(x, y, z)
    canvas.draw()


############gui##############
window = tk.Tk()
window.title('perceptron')
window.geometry('1000x1000')

# model parameter
learning_rate = tk.DoubleVar()
entry_learning_rate = tk.Entry(window, textvariable=learning_rate)
entry_learning_rate.place(x=160, y=50)
tk.Label(window, text='learning rate: ').place(x=50, y= 50)
iteration = tk.IntVar()
entry_iteration = tk.Entry(window, textvariable=iteration)
entry_iteration.place(x=160, y=90)
tk.Label(window, text='iteration: ').place(x=50, y=90)

#select data
dataset_url = ""
filename = tk.StringVar()
selected_file = tk.Label(window, textvariable=filename).place(x=280, y= 10)
select_data_btn = tk.Button(window, text='select dataset', command=openfile).place(x=160, y=10)

#start
btn_start = tk.Button(window, text='start', command=start)
btn_start.place(x=50, y=130)

#weights
tk.Label(window, text='trained_weights: ').place(x=50, y= 170)
trained_weights = tk.StringVar()
lb_trained_weights = tk.Label(window, textvariable=trained_weights).place(x=160, y=170)

#score
tk.Label(window, text='train score: ').place(x=50, y= 210)
trained_score = tk.StringVar()
lb_trained_score = tk.Label(window, textvariable=trained_score).place(x=160, y=210)

tk.Label(window, text='test score: ').place(x=50, y= 250)
test_score = tk.StringVar()
lb_trained_score = tk.Label(window, textvariable=test_score).place(x=160, y=250)

#plot
# the figure that will contain the plot
fig = Figure(figsize = (5, 5),
        dpi = 100)

# adding the subplot

# creating the Tkinter canvas
# containing the Matplotlib figure
canvas = FigureCanvasTkAgg(fig,
                            master = window) 

# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas,
                                window)
toolbar.update()

# placing the toolbar on the Tkinter window
canvas.get_tk_widget().place(x=160, y=290)

window.mainloop()









