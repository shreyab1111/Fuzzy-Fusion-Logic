import pandas as pd
import numpy as np

rank=pd.read_csv('./3_mod/rank.csv')
clp=pd.read_csv('./3_mod/ClassPrediction.csv')
confi=pd.read_csv('./3_mod/confidence.csv')


print(rank.columns)
print(clp.columns)
print(confi.columns)

H=3
pr=0.25
pc=0.0
ypred=[]
for i in range(129):
    rs0=0
    rs1=0
    cf0=0
    cf1=0
    ls=[]
    # for class 0==========>>>>>>>>>>>>>
    ### for vgg16
    if clp['vgg16'][i]==0:
        rs0+=rank['vgg16_0'][i]
        cf0+=confi['vgg16_0'][i]
    else:
        rs0+=pr
        cf0+=pc
    ### for vgg19
    if clp['vgg19'][i]==0:
        rs0+=rank['vgg19_0'][i]
        cf0+=confi['vgg19_0'][i]
    else:
        rs0+=pr
        cf0+=pc
    ### for xception
    if clp['xcep'][i]==0:
        rs0+=rank['xcep_0'][i]
        cf0+=confi['xcep_0'][i]
    else:
        rs0+=pr
        cf0+=pc
        
    cf0=1-(cf0/H)
    
    # for class 1==========>>>>>>>>>>>>>
    ### for vgg16
    if clp['vgg16'][i]==0:
        rs1+=rank['vgg16_1'][i]
        cf1+=confi['vgg16_1'][i]
    else:
        rs1+=pr
        cf1+=pc
    ### for vgg19
    if clp['vgg19'][i]==0:
        rs1+=rank['vgg19_1'][i]
        cf1+=confi['vgg19_1'][i]
    else:
        rs1+=pr
        cf1+=pc
    ### for xception
    if clp['xcep'][i]==0:
        rs1+=rank['xcep_1'][i]
        cf1+=confi['xcep_1'][i]
    else:
        rs1+=pr
        cf1+=pc
        
    cf1=1-(cf1/H)
    ls.append(rs0*cf0)
    ls.append(rs1*cf1)
    ypred.append(np.argmin(ls))


print(ypred)
print(len(ypred))

ytest=[]
for i in range(1,130):
    if i<=43:
        ytest.append(0)
    else:
        ytest.append(1)
        
print(len(ytest))

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
print("Accuracy of the model is: ",accuracy_score(ytest,ypred)*100)
print("Classification Report is:\n",classification_report(ytest,ypred))
print("The confusion matrix is: \n",confusion_matrix(ytest,ypred))