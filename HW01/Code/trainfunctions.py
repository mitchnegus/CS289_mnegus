# trainfunctions.py
#------------------------------------------
# Python module for CS289A HW01
#------------------------------------------
#------------------------------------------


from sklearn import svm


def TrainAndScoreNsamples(trainsetarrays,trainsetlabels,valsetarrays,valsetlabels,hyperparam):
# Train the classifier and score on the validation set
    clf = svm.SVC(C=hyperparam,kernel='linear')
    clf.fit(trainsetarrays,trainsetlabels)
    return clf.score(valsetarrays,valsetlabels)
    
    
def TrainAndPredictNsamples(trainsetarrays,trainsetlabels,testarrays,hyperparam):
# Train the classifier and predict on the test array
    clf = svm.SVC(C=hyperparam,kernel='linear')
    clf.fit(trainsetarrays,trainsetlabels)
    return clf.predict(testarrays)