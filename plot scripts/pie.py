import matplotlib.pyplot as plt
import numpy as np

def make_pie_chart():
    labels = 'Games', 'Potato Seeds', 'Cell Phone', 'Gas', 'Fertilizer'
    colors = ['r', 'g', '#ff42a5f5', 'c', 'm']
    # in our case our much we spent per month on each item in label 
    data = [500, 200, 30, 200, 150]
    explode = (0.1, 0, 0, 0, 0)
    plt.pie(data, labels=labels, colors=colors, explode=explode, startangle=90, shadow=True, autopct='%1.1f%%')
    plt.show()

make_pie_chart()