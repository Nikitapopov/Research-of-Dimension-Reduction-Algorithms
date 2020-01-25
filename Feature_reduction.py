import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, SelectFromModel, chi2, SelectFpr, SelectFdr, RFE, RFECV,VarianceThreshold,SelectKBest
from sklearn.svm import SVR, LinearSVC, LinearSVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score,confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time

def myVarianceThreshold(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    start_time = time.time()
    finish_time=time.time()
    print("time = ", finish_time-start_time)
    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    start_time = time.time()
    selector = VarianceThreshold(threshold=0.005)
    X_new_train = X_train[X_train.columns[selector.get_support(indices=True)]]
    finish_time=time.time()
    print("time = ", finish_time-start_time)
    X_new_test = X_test[X_test.columns[selector.get_support(indices=True)]]
    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_new_train, y_train)
    rfc_prediction = rfc_model.predict(X_new_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    return X

def mySelectKBest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    fit = SelectKBest(chi2, k=10).fit(X_train, y_train)
    finish_time=time.time()
    print("time = ", finish_time-start_time)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores)

    acc_score = [0]*16
    for i in range(1,17):
        fit = SelectKBest(score_func=chi2, k=i).fit(X_train,y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train_new, y_train)
        prediction = model.predict(X_test_new)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy")
    plt.show()

def myPCA(X, y):
    start_time = time.time()
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = PCA(n_components=i).fit(X)
        X_new = pd.DataFrame(fit.transform(X))
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=27)
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy")
    plt.show()

def mySelectFromModel_LR(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    log_reg_model = LogisticRegression(solver='lbfgs')
    smf = SelectFromModel(log_reg_model)
    smf.fit(X_train, y_train)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        smf = SelectFromModel(log_reg_model, max_features=i, threshold=-np.inf)
        smf.fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[smf.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[smf.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])
    return acc_score

def myRFE_LR(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    log_reg_model = LogisticRegression(solver='lbfgs')
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = RFE(log_reg_model, i).fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features:", i, ". Accuracy:", acc_score[i -1])
    return acc_score

def mySelectFromModel_RFC(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    ram_for_cl_model = RandomForestClassifier(n_estimators = 30)
    smf = SelectFromModel(ram_for_cl_model)
    smf.fit(X_train, y_train)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        smf = SelectFromModel(ram_for_cl_model, max_features=i, threshold=-np.inf)
        smf.fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[smf.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[smf.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])
    return acc_score

def myRFE_RFC(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    ram_for_cl_model = RandomForestClassifier(n_estimators=30)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = RFE(ram_for_cl_model, i).fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features:", i, ". Accuracy:", acc_score[i - 1])
    return acc_score

def mySelectFromModel_ETC(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    ext_tr_cl_model = ExtraTreesClassifier(n_estimators = 30)
    smf = SelectFromModel(ext_tr_cl_model)
    smf.fit(X_train, y_train)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        smf = SelectFromModel(ext_tr_cl_model, max_features=i, threshold=-np.inf)
        smf.fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[smf.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[smf.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])
    return acc_score

def myRFE_ETC(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    ext_tr_cl_model = ExtraTreesClassifier(n_estimators=30)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = RFE(ext_tr_cl_model, i).fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features:", i, ". Accuracy:", acc_score[i - 1])
    return acc_score

def diagramOfMethods(title, model1, model2, model3 = 'null'):
    plt.figure(1, figsize=(14, 13))
    plt.clf()
    features = [i+1 for i in range(16)]
    plt.plot(features, model1, 'x-')
    plt.plot(features, model2, '*-')
    if model3 != 'null':
        plt.plot(features, model3, 'p-')
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    if model3 != 'null':
        plt.legend(['LogisticRegression', 'RandomForestClassifier', 'ExtraTreesClassifier'])
    else:
        plt.legend(['SelectFromModel', 'RFE'])
    plt.title("Accuracy of  " + title)
    plt.show()

def preparationData():
    data = pd.read_csv('data/data1_train.csv')

    data = data.drop(['ID'], axis=1).loc[data['pdays'] != -1]

    data['subscribed'] = data['subscribed'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['job'] = data['job'].replace(to_replace=['admin.', 'entrepreneur', 'blue-collar', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], value=[1,2,3,4,5,6,7,8,9,10,11,0])
    data['marital'] = data['marital'].replace(to_replace=['divorced', 'married', 'single'], value=[1,2,3])
    data['education'] = data['education'].replace(to_replace=['primary', 'secondary', 'tertiary', 'unknown'], value=[1, 2, 3, 0])
    data['default'] = data['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['housing'] = data['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['loan'] = data['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['contact'] = data['contact'].replace(to_replace=['cellular', 'telephone', 'unknown'], value=[1, 2, 0])
    data['month'] = data['month'].replace(to_replace=['apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'], value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    data['poutcome'] = data['poutcome'].replace(to_replace=['failure', 'success', 'other', 'unknown'], value=[1, 2, 3, 0])

    data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
    X_nm = data_nm.drop(['subscribed'], axis=1)
    Y_nm = data_nm['subscribed']
    X_nm.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    return X, Y, X_nm, Y_nm

if __name__ == '__main__':
    x_nm, y_nm = preparationData()

    myVarianceThreshold(x_nm, y_nm)
    mySelectKBest(x_nm, y_nm)
    myPCA(x_nm, y_nm)

    #Logical regression
    acc_score_SFM_LR = mySelectFromModel_LR(x_nm, y_nm)
    acc_score_RFE_LR = myRFE_LR(x_nm, y_nm)

    #Random forest classification
    acc_score_SFM_RFC = mySelectFromModel_RFC(x_nm, y_nm)
    acc_score_RFE_RFC = myRFE_RFC(x_nm, y_nm)

    #Extra trees clas
    acc_score_SFM_ETC = mySelectFromModel_ETC(x_nm, y_nm)
    acc_score_RFE_ETC = myRFE_ETC(x_nm, y_nm)

    #Comparison SFM
    diagramOfMethods('method SelectFromModel', acc_score_SFM_LR, acc_score_SFM_RFC, acc_score_SFM_ETC)
    diagramOfMethods('method RFE', acc_score_RFE_LR, acc_score_RFE_RFC, acc_score_RFE_ETC)
    diagramOfMethods('model logistic regression', acc_score_SFM_RFC, acc_score_RFE_RFC)

    summ1 = 0
    summ2 = 0
    summ3 = 0
    summ4 = 0
    summ5 = 0
    for i in range(16):
        summ1 += acc_score_SFM_LR[i] + acc_score_RFE_LR[i]
        summ2 += acc_score_SFM_RFC[i] + acc_score_RFE_RFC[i]
        summ3 += acc_score_SFM_ETC[i] + acc_score_RFE_ETC[i]
        summ4 += acc_score_SFM_ETC[i]
        summ5 += acc_score_RFE_ETC[i]
    print(summ1/summ2)
    print(summ1/summ3)
    print(summ4/summ5)