#KNN CLASSIFICATION

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x=[4,5,10,4,3,11,14,8,10,12]
y=[21,19,24,17,16,25,24,22,21,21]
classes=[0,0,1,0,0,1,1,0,1,1]
data=list(zip(x,y))
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(data,classes)
new_x=14
new_y=21
new_point=[(new_x,new_y)]
prediction=knn.predict(new_point)
print(prediction)
plt.scatter(x+[new_x],y+[new_y],c=classes+[prediction[0]])
plt.text(x=new_x-1.7,y=new_y-0.7,s=f"newpoint,class:{prediction[0]}")
plt.show()            
            
