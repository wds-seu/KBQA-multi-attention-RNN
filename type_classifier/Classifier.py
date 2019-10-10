from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing


# 2分类 simple complex分类
target_names = ['class 0', 'class 1']

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 8))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:8]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def print_type_result(y_test , pre_y_test):
    num = 0
    right = 0
    i_type = ''
    j_type = ''
    for _ in arange(1000):
        i = y_test[num]
        j = pre_y_test[num]

        if str(i) in {'0', '2', '4'}:
            i_type = 'simple'
        else:
            i_type = 'complex'

        if str(j) in {'0', '2', '4'}:
            j_type = 'simple'
        else:
            j_type = 'complex'

        if i_type == j_type:
            right += 1
            #print(str(num) + "," + str(i) + "," + str(j) + "," + "type_true")
            print("type_true")
        else:
            #print(str(num) + "," + str(i) + "," + str(j) + "," + "type_false")
            print("type_false")
        num += 1
    print(right)


def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = knn.predict(X_test)

    print(" KNN Classifier")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))


def Multinomial_Naive_Bayes(X_train, X_test, y_train, y_test):
    bayes = MultinomialNB()
    bayes.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = bayes.predict(X_test)

    print("Multinomial Naive Bayes Classifier")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))


def Logistic_Regression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(penalty='l1')
    lr.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = lr.predict(X_test)

    print("Logistic Regression Classifier")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))
    print_type_result(y_test, pre_y_test)




def Random_Forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=8)
    rf.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = rf.predict(X_test)

    print("Random Forest Classifier ")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, list(pre_y_test)))


def Decision_Tree(X_train, X_test, y_train, y_test):
    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = dt.predict(X_test)

    print("Decision Tree Classifier ")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))


def GBDT(X_train, X_test, y_train, y_test):
    gbdt = GradientBoostingClassifier(n_estimators=200)
    gbdt.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = gbdt.predict(X_test)

    print("GBDT(Gradient Boosting Decision Tree) Classifier ")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))


def SVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = svm.predict(X_test)

    print("SVM Classifier")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))


def MLP(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    mlp.fit(X_train, y_train)
    # 预测值 pre_y_test 真实值 y_test
    pre_y_test = mlp.predict(X_test)

    print("MLPClassifier")
    print(classification_report(y_test, pre_y_test, target_names=target_names))
    print("accuracy_score:")
    print(accuracy_score(y_test, pre_y_test))



if __name__ == "__main__" :


    datingDataMat_train, datingLabels_train = file2matrix('type2_train.txt')
    datingDataMat_test, datingLabels_test = file2matrix('type2_test.txt')

    # 预处理
    '''
    # z-score标准化
    datingDataMat = preprocessing.scale(datingDataMat_train)
    datingDataMat = preprocessing.scale(datingDataMat_test)
    '''
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    datingDataMat = min_max_scaler.fit_transform(datingDataMat)
    '''
    '''
    max_abs_scaler = preprocessing.MaxAbsScaler()
    datingDataMat = max_abs_scaler.fit_transform(datingDataMat)
    '''
    X_train = datingDataMat_train
    X_test = datingDataMat_test
    y_train = datingLabels_train
    y_test = datingLabels_test

    KNN(X_train, X_test, y_train, y_test)
    Multinomial_Naive_Bayes(X_train, X_test, y_train, y_test)
    Logistic_Regression(X_train, X_test, y_train, y_test)
    Random_Forest(X_train, X_test, y_train, y_test)
    Decision_Tree(X_train, X_test, y_train, y_test)
    GBDT(X_train, X_test, y_train, y_test)
    SVM(X_train, X_test, y_train, y_test)
    MLP(X_train, X_test, y_train, y_test)
