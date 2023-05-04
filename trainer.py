import os
import soundfile as sf
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint

import time
import pickle

START = time.time()
start = time.time()

directory = os.scandir('./res/music/wav')
X = []
y = []
start = time.time()
for file in directory:
    songTime = time.time()
    x,fs = sf.read(file.path)


    n = 0
    #for each song we're going to take 10 samples per song, starting at 30s until 165s in 15s lapses
    for i in range(30,180,15):

        # We're only going to take the first channel between 30 and 35 seconds, 60 and 90 and 95
        if(x.shape[0]>(i+5)*fs): #Checking that we're not sampling away of the song's end
            n += 1
            xTaken = x[i*fs:(i+5)*fs].copy().T[0]
            xfourier = np.fft.rfft(xTaken)

            X.append(abs(xfourier))
            y.append(file.name.split('-')[0])

    print(file.name[:-4],":",round(time.time()-songTime,3),"s, ",n,"samples")

end = time.time()

print("\nCollected",len(X),"samples in:",round(end-start,3),"s")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

param_dist = {
    'n_estimators': randint(50,500),
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': randint(1,30)
}

rf = RandomForestClassifier()

rs = RandomizedSearchCV(rf,param_dist,n_iter=10,verbose=3)

rs.fit(X_train,y_train)

end = time.time()

print("Training Random Forest with randomized search cross validation: ",end-start," s")

best_rf = rs.best_estimator_

print("\nbest random forest hyperparameters",rs.best_params_)

y_pred = best_rf.predict(X_test)

print("Accuracy",accuracy_score(y_test,y_pred))

print("Saving model")
pickle.dump(best_rf,open('./res/models/randomForest.pickle','wb'))



print("Total process time: ",time.time()-START," s")

