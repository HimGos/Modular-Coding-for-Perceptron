import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def prepare_data(df, target_column="y"):
    X = df.drop(target_column, axis=1)
    
    y = df[target_column]
    
    return X, y


def save_plot(df, model, filename="plot.png", plot_dir = "plots"):
    
    #Internal function
    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c='y', s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)  #Horizontal line
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)  #Vertical Line
        
        figure = plt.gcf()
        figure.set_size_inches(10,8)
        
    def _plot_decision_regions(X, y, classifier, resolution=0.02):        #Resolution makes graph look continuous
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)
        
        X = X.values   #As an array
        x1 = X[:, 0]
        x2 = X[:, 0]
        
        #creating space between units of graph
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        
        #Creating Mesh Grid
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                              np.arange(x2_min, x2_max, resolution)
                              )
        
        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)         #Ravel flattens array
        y_hat = y_hat.reshape(xx1.shape)   # Matrix form
        
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.xlim(xx2.min(), xx2.max())
        
        plt.plot()
        
    X, y = prepare_data(df)
    
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)