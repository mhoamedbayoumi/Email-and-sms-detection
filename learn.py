# this program will ditect if email spam(1) or not spam(0)

#imporing important modules that will need 
import re
import tkinter as tk
from PIL import Image,ImageTk
from numpy.lib.shape_base import split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# read the file 
df=pd.read_csv("spam.csv", encoding='latin-1')
print(df.head())
  

feature=df['v2']     # out text email or sms we want to train
label=df['v1']     # what we want to predict 

print(df.head())


# split our data to train data and test data 
X_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=7)

#make object from tfidvectotizer type words english max words 7
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# train the module on the training data 
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
clf=SVC()
#trian y dataset

clf.fit(tfidf_train,y_train)
# what we will pridect 

y_pred=clf.predict(tfidf_test)
#the acurrcay score
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

##############################################
# user=[input()]
# user_transform=tfidf_vectorizer.transform(user)
# print(clf.predict(user_transform))
######################################################
root=tk.Tk()

root.geometry("350x400")
root.title("email detiction")

root.configure(background='#33b5ff')
main_text=tk.Label(root,text="WeLcome to email Detection",font=('arial',16),fg='white',bg='#33b5ff').pack(padx=5)

user_entry=tk.Entry(root)
user_entry.insert(0,"enter the email")
user_entry.pack(pady=45)
def reslut():
    us=[user_entry.get()]
    user_transform=tfidf_vectorizer.transform(us)
    res=clf.predict(user_transform)
    if res==['ham']:
        tk.Label(root,text="not spam email").pack(pady=10)
    else:
        tk.Label(root,text="spam email").pack(pady=10)
btn=tk.Button(root,text="Enter",command=reslut,font=("arial",10),bg="white",fg="black")
def on_enter(e):
    btn['background'] = '#005c91'

def on_leave(e):
    btn['background'] = 'white'
btn.bind("<Enter>",on_enter)
btn.bind("<Leave>",on_leave)
btn.pack()
btn.pack(expand=True,pady=10)
root.mainloop()
