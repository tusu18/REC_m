#!/usr/bin/env python
# coding: utf-8

# In[246]:


import pandas as pd
import numpy as np
from flask import Flask, render_template, request


# In[206]:


df1=pd.read_csv("sih_dummy.csv")


# In[135]:


df1["DOMAIN"]=df1["DOMAIN"].str.replace(" ","")


# In[209]:


df1["DOMAIN"]=df1["DOMAIN"].str.lower()


# In[210]:


df1["EXPERIENCE"]=df1["EXPERIENCE"].astype(str)


# In[211]:


df1["CITY"]=df1["CITY"].str.lower()


# In[232]:


df1["joint"]= df1[["DOMAIN","EXPERIENCE"]].apply(lambda x: "".join(x),axis=1)


# In[234]:


df1["joint"]=df1[["joint","CITY"]].apply(lambda x: " ".join(x),axis=1)


# In[236]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[237]:


cv=TfidfVectorizer()


# In[238]:


cnt_mat=cv.fit_transform(df1["joint"])


# In[239]:


sim=cosine_similarity(cnt_mat)


# In[241]:


def rcom(m):
    m=m.lower()
    if m not in df1["DOMAIN"].unique():
        print("Out of Domain")
    else:
        i=df1.loc[df1["DOMAIN"] == m].index[0]
        lst=list(enumerate(sim[i]))
        lst=sorted(lst,key=lambda x:x[1],reverse=True)
        lst=lst[1:5]
        l=[]
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(df1["NAME"][a])
        return l
        


# In[247]:


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    domain = [int(x) for x in request.form.values()]
    r = [np.array(domain)]
    rcmd=rcom(r)
    return render_template('index.html',prediction_text="Thus ".format(rcmd))

# In[ ]:


if __name__ == '__main__':
    app.run()


# In[ ]:





# In[ ]:




