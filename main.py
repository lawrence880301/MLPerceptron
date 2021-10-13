from DataPreprocessor import Datapreprocessor
from MLPerceptron import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tkinter as tk
from tkinter import filedialog
import random



def openfile():
    global dataset_url, filename
    dataset_url = filedialog.askopenfilename(title="Select file")
    filename = filename.split("/")[-1]
    filename = filename.split(".")[0]
    return dataset_url

def prepare_data():
    dataset = Datapreprocessor.readfile(dataset_url)
    dataset = Datapreprocessor.text_to_numlist(dataset)
    label_count = len(Datapreprocessor.label_list(dataset))
    feature_count = Datapreprocessor.num_of_feature(dataset)
    train_data, test_data = Datapreprocessor.train_test_split(dataset, 2/3)
    train_x, train_y = Datapreprocessor.feature_label_split(train_data)
    test_x, test_y = Datapreprocessor.feature_label_split(test_data)
    return train_x, train_y, test_x, test_y, label_count, feature_count



#gui
window = tk.Tk()
window.title('multilayer-perceptron')
window.geometry('600x600')

# welcome image
canvas = tk.Canvas(window, height=400, width=400)
# image_file = tk.PhotoImage(file='welcome.gif')
# image = canvas.create_image(0,0, anchor='nw', image=image_file)
canvas.pack(side='top')

# model parameter
neurons_per_layer = tk.IntVar()
entry_neurons_per_layer = tk.Entry(window, textvariable=neurons_per_layer)
entry_neurons_per_layer.place(x=160, y=150)
tk.Label(window, text='neurons per layer: ').place(x=50, y= 150)
num_layer = tk.IntVar()
entry_num_layer = tk.Entry(window, textvariable=num_layer)
entry_num_layer.place(x=160, y=190)
tk.Label(window, text='number of layers: ').place(x=50, y= 190)
learning_rate = tk.DoubleVar()
entry_learning_rate = tk.Entry(window, textvariable=learning_rate)
entry_learning_rate.place(x=160, y=230)
tk.Label(window, text='learning rate: ').place(x=50, y= 230)
iteration = tk.IntVar()
entry_iteration = tk.Entry(window, textvariable=iteration)
entry_iteration.place(x=160, y=270)
tk.Label(window, text='iteration: ').place(x=50, y= 270)


def start():
    train_x, train_y, test_x, test_y , label_count, feature_count= prepare_data()
    modified_train_y = Datapreprocessor.label_preprocess(train_y)
    modified_test_y = Datapreprocessor.label_preprocess(test_y)
    model = MLPerceptron(num_layer.get(),neurons_per_layer.get(),feature_count,2,learning_rate.get())
    losses = model.train(train_x, modified_train_y, iteration.get())
    print(model.predict([0,0]))

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('losses')
    plt.show()

dataset_url = ""
filename = ""
tk.Label(window, textvariable=filename).place(x=250, y= 110)
select_data_btn = tk.Button(window, text='select dataset', command=openfile)
select_data_btn.place(x=160, y=110)
btn_login = tk.Button(window, text='start', command=start)
btn_login.place(x=160, y=310)
window.mainloop()






# model = MLPerceptron(0,2,2,1,0.1)
# model.train(train_x, train_y, 100)
# print(model.predict(test_x))
# print(test_y)
# print(model.layers[1].weights)


