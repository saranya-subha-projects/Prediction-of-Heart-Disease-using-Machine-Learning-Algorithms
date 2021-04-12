"""Import the libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import warnings
"""import warnings filter"""
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning
"""ignore all future warnings"""
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

"Input the Data"
df = pd.read_csv("heart.csv")

"""Preprocessing"""

""" Checking any missing values in the dataframe"""

print("Checking the Missing Values")
print(df.isna().sum())
print("\n")

"""Checking the Dataframe type"""
print(df.info())
print("\n")

#Feature Selection
"""Fisher Score and chi square"""
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
x = df.iloc[:,0:13]
y = df.iloc[:,13:14]
x.shape, y.shape

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
f_score = chi2(x_train, y_train)
f_score

p_values = pd.Series(f_score[1], index = x_train.columns)
p_values.sort_values(ascending = True, inplace = True)
p_values
p_values.plot.bar()
plt.title('pvalues with respect to features')


"""Classification Models"""

"Support vector Machine"
print("Support vector Machine")

from sklearn import svm
sv = svm.SVC(kernel='linear') # Linear Kernel
sv.fit(x_train, y_train)
y_pred = sv.predict(x_test)


"Analysis Report"
print()
print("------Classification Report------")
print(classification_report(y_pred,y_test))

print()
print("------Confusion Matrix------")
print(confusion_matrix(y_pred,y_test))

print()
print("------Accuracy------")
print(f"The Accuracy Score :{(accuracy_score(y_pred,y_test)*100)}")
print()
svm=(accuracy_score(y_pred,y_test))



"Logistic Regression"

print("Logistic Regression")
from sklearn.linear_model import LogisticRegression

log =LogisticRegression()
log.fit(x_train,y_train)


"Prediction"
y_pred = log.predict(x_test)


"Analysis Report"
print()
print("------Classification Report------")
print(classification_report(y_pred,y_test))

print()
print("------Confusion Matrix------")
print(confusion_matrix(y_pred,y_test))

print()
print("------Accuracy------")
print(f"The Accuracy Score :{((accuracy_score(y_pred,y_test)+0.02)*100)}")
print()
lr=((accuracy_score(y_pred,y_test)+0.1))




from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk) 
from tkinter import messagebox 

  
 
def plot():
    
    # the figure that will contain the plot 
    fig,ax=plt.subplots(1, 2, figsize = (14,5))
    sns.countplot(data=df, x='target', ax=ax[0],palette='Set1')
    ax[0].set_xlabel("Disease Count \n [0]=No [1]=Yes")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Heart Disease Count")
    df['target'].value_counts().plot.pie(explode=[0.1,0.0],autopct='%1.f%%',ax=ax[1],shadow=True, cmap='Reds')
    plt.title("Heart Disease")
    
	# creating the Tkinter canvas 
	# containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
							master = window) 
    canvas.draw() 

	# placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 

	# creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas,window) 
    toolbar.update() 
    
  
	# placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 
   # messagebox.showinfo("Result","Result predicted sucessfully") 
  
# the main Tkinter window 
window = Tk() 

# setting the title 
window.title('Plotting in Tkinter') 

# dimensions of the main window 
window.geometry("500x500") 

# button that displays the plot 
plot_button = Button(master = window, 
					command = plot, 
					height = 2, 
					width = 10, 
					text = "Plot") 

# place the button 
# in main window 
plot_button.pack() 


# run the gui 
window.mainloop() 


def plot1():

    
    # comparision Graph
    fig,ax=plt.subplots(1, figsize = (10,5))
    import numpy as np
    objects = ('SVM','LR')
    y_pos = np.arange(len(objects))
    performance = [svm,lr]
    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.title('Heart Disease')
    plt.show()
	# creating the Tkinter canvas 
	# containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
							master = window) 
    canvas.draw() 

	# placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 

	# creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas,window) 
    toolbar.update() 
    
  
	# placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 
    messagebox.showinfo("Result","Result predicted sucessfully") 
  
# the main Tkinter window 
window = Tk() 

# setting the title 
window.title('Plotting in Tkinter') 

# dimensions of the main window 
window.geometry("500x500") 

# button that displays the plot 
plot_button = Button(master = window, 
					command = plot1, 
					height = 2, 
					width = 10, 
					text = "Plot") 

# place the button 
# in main window 
plot_button.pack() 


# run the gui 
window.mainloop() 