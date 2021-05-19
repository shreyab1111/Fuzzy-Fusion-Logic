from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import math as m

## this func calculates the rank
def rank_cal(n):
    return (1-m.exp((n-1.0)*(1.0-n)/2.0))


confidence=[]      # this will store all the confidence of the models
rank=[]            # this will store the calculated rank
def RankConfidence(model1,model2,model3):
    test_no='./TEST/NO'
    test_yes='./TEST/YES'
    for file in os.listdir(test_no):
        indi_conf=[]
        indi_rank=[]
        res=[]
        img=image.load_img(os.path.join(test_no,file),target_size=(224,224,3))
        img = image.img_to_array(img)
        img=img/255.0
        img = np.expand_dims(img, axis=0)
        res1=model1.predict(img,batch_size=1)
        res2=model2.predict(img,batch_size=1)
        res3=model3.predict(img,batch_size=1)
        #res4=model4.predict(img,batch_size=1)
        #res5=model5.predict(img,batch_size=1)
        res.append(res1[0])
        res.append(res2[0])
        res.append(res3[0])
        #res.append(res4[0])
        #res.append(res5[0])
        for i in res:
            for j in i:
                indi_conf.append(j)
                indi_rank.append(rank_cal(j))
        confidence.append(indi_conf)
        rank.append(indi_rank)
        
    ### end of the loop
    #print(ypred)
    for file in os.listdir(test_yes):
        indi_conf=[]
        indi_rank=[]
        res=[]
        img=image.load_img(os.path.join(test_yes,file),target_size=(224,224,3))
        img = image.img_to_array(img)
        img=img/255.0
        img = np.expand_dims(img, axis=0)
        res1=model1.predict(img,batch_size=1)
        res2=model2.predict(img,batch_size=1)
        res3=model3.predict(img,batch_size=1)
        #res4=model4.predict(img,batch_size=1)
        #res5=model5.predict(img,batch_size=1)
        res.append(res1[0])
        res.append(res2[0])
        res.append(res3[0])
        #res.append(res4[0])
        #res.append(res5[0])
        for i in res:
            for j in i:
                indi_conf.append(j)
                indi_rank.append(rank_cal(j))
        confidence.append(indi_conf)
        rank.append(indi_rank)


model1=load_model('./modelNew/vgg16_500.h5')
model2=load_model('./modelNew/vgg19_500.h5')
model3=load_model('./modelNew/xcep_500.h5')
#model4=load_model('./modelNew/res50_500.h5')
#model5=load_model('./modelNew/incv3_500.h5')

RankConfidence(model1,model2,model3)


print("Confidence matrix is: \n")
for i in range(5):
    print(confidence[i])

print("Rank matrix is: \n")
for i in range(5):
    print(rank[i])


# change the array to numpy array
confidence=np.asarray(confidence)
rank=np.asarray(rank)

pd.DataFrame(rank).to_csv('./3_mod/rank.csv',header=head,index=None)

data=pd.read_csv('./3_mod/rank.csv')
data.head()
print(data.info())

classPrediction=[]
def topClass(model1,model2,model3):
    test_no='./TEST/NO'
    test_yes='./TEST/YES'
    for file in os.listdir(test_no):
        res=[]
        temp=[]
        img=image.load_img(os.path.join(test_no,file),target_size=(224,224,3))
        img = image.img_to_array(img)
        img=img/255.0
        img = np.expand_dims(img, axis=0)
        res1=model1.predict(img,batch_size=1)
        res2=model2.predict(img,batch_size=1)
        res3=model3.predict(img,batch_size=1)
        #res4=model4.predict(img,batch_size=1)
        #res5=model5.predict(img,batch_size=1)
        res.append(res1[0])
        res.append(res2[0])
        res.append(res3[0])
        #res.append(res4[0])
        #res.append(res5[0])
        for i in res:
            if i[0]>i[1]:
                temp.append(0)
            else:
                temp.append(1)
    
        classPrediction.append(temp)
        
    ### end of the loop
    #print(ypred)
    for file in os.listdir(test_yes):
        temp=[]
        res=[]
        img=image.load_img(os.path.join(test_yes,file),target_size=(224,224,3))
        img = image.img_to_array(img)
        img=img/255.0
        img = np.expand_dims(img, axis=0)
        res1=model1.predict(img,batch_size=1)
        res2=model2.predict(img,batch_size=1)
        res3=model3.predict(img,batch_size=1)
        #res4=model4.predict(img,batch_size=1)
        #res5=model5.predict(img,batch_size=1)
        res.append(res1[0])
        res.append(res2[0])
        res.append(res3[0])
        #res.append(res4[0])
        #res.append(res5[0])
        for i in res:
            if i[0]>i[1]:
                temp.append(0)
            else:
                temp.append(1)
    
        classPrediction.append(temp)


topClass(model1,model2,model3)

print("Class prediction:\n")
for i in range(5):
    print(classPrediction[i])


classPrediction=np.array(classPrediction)
head=['vgg16','vgg19','xcep']
pd.DataFrame(classPrediction).to_csv('./3_mod/ClassPrediction.csv',header=head,index=None)

data=pd.read_csv('./3_mod/ClassPrediction.csv')
data.head()

data=pd.read_csv('./3_mod/rank.csv')
data.head()

