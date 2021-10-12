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
    filename = dataset_url.split("/")[-1]
    filename = filename.split(".")[0]
    return dataset_url

def prepare_data(url):
    dataset = Datapreprocessor.readfile(url)
    dataset = Datapreprocessor.text_to_numlist(dataset)
    train_data, test_data = Datapreprocessor.train_test_split(dataset, 2/3)
    train_x, train_y = Datapreprocessor.feature_label_split(train_data)
    test_x, test_y = Datapreprocessor.feature_label_split(test_data)
    return train_x, train_y, test_x, test_y



##gui
window = tk.Tk()
window.title("")
window.geometry('750x450')

# select dataset
label_data = tk.Label(window, text="select dataset", fg='black')
label_data.place(x=30, y=10)
button_data = tk.Button(window, text='select', fg='blue', command=openfile)
button_data.place(x=120, y=10)

# input lr
label_lr = tk.Label(window, text="learning rate", fg='black')
label_lr.place(x=30, y=60)
input_lr = tk.DoubleVar()
input_lr.set(0.8)
entry_lr = tk.Entry(window, width=20, textvariable=input_lr)
entry_lr.place(x=120, y=60)

# input epoch
label_epoch = tk.Label(window, text="epoch", fg='black')
label_epoch.place(x=30, y=90)
input_epoch = tk.IntVar()
input_epoch.set(1)
entry_epoch = tk.Entry(window, width=20, textvariable=input_epoch)
entry_epoch .place(x=120, y=90)

# display weight
label_weight0 = tk.Label(window, text="weight0", fg='blue')
label_weight0.place(x=30, y=140)
label_weight1 = tk.Label(window, text="weight1", fg='blue')
label_weight1.place(x=30, y=170)
label_weight2 = tk.Label(window, text="weight2", fg='blue')
label_weight2.place(x=30, y=200)

# display accuracy
label_train_acc = tk.Label(window, text="training_acc", fg='blue')
label_train_acc.place(x=30, y=250)
label_test_acc = tk.Label(window, text="testing_acc", fg='blue')
label_test_acc.place(x=30, y=280)

f = Figure(figsize=(3, 3), dpi=150)
f.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

f_plot = f.add_subplot(111)
f_plot.tick_params(labelsize=8)
canvas = FigureCanvasTkAgg(f, window)
canvas.get_tk_widget().place(x=300, y=0)

# submit botton
button = tk.Button(window, height=2, width=8, text='run', fg='blue', command=perceptron)
button.place(x=120, y=350)

window.mainloop()


# n_hiddenlayers, n_neuron, n_input, n_output, lr

model = MLPerceptron(0,2,2,1,0.1)
model.train(train_x, train_y, 100)
print(model.predict(test_x))
print(test_y)
print(model.layers[1].weights)

x = np.array(range(2))
y = -(model.layers[1].weights[0]*x + model.layers[1].bias)/model.layers[1].weights[1]
plt.plot(x, y ,linewidth = 1, color = 'black') #畫圖 ms：折點大小

plt.scatter(train_x[0], train_x[1])
plt.show()
