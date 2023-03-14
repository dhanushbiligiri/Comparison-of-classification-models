import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2

#Read the data
data=pd.read_csv('SMSSpamCollection.csv', sep='\t',names=["label", "message"])

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# feature extraction using count vectorizer
cv = CountVectorizer()
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

# apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_counts.toarray())
X_test_pca = pca.transform(X_test_counts.toarray())

# apply RFE
lr = LogisticRegression()
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(X_train_counts, y_train)
selected_features = [list(cv.vocabulary_.keys())[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
X_train_rfe = rfe.transform(X_train_counts)
X_test_rfe = rfe.transform(X_test_counts)

# feature extraction using chi-squared test
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
selector = SelectKBest(chi2, k=1000)
X_train_chi2 = selector.fit_transform(X_train_counts, y_train)
X_test_chi2 = selector.transform(X_test_counts)

# evaluate different classification methods
models = {
    #'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    print(f'--- {name} ---')
    
    # using PCA
    model.fit(X_train_pca, y_train)
    y_pred_pca = model.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    print(f'Accuracy using PCA: {acc_pca}')
    print(f'Classification report using PCA:\n{classification_report(y_test, y_pred_pca)}')
    print(f'Confusion matrix using PCA:\n{confusion_matrix(y_test, y_pred_pca)}')
    
    # using RFE
    model.fit(X_train_rfe, y_train)
    y_pred_rfe = model.predict(X_test_rfe)
    acc_rfe = accuracy_score(y_test, y_pred_rfe)
    print(f'Accuracy using RFE: {acc_rfe}')
    print(f'Classification report using RFE:\n{classification_report(y_test, y_pred_rfe)}')
    print(f'Confusion matrix using RFE:\n{confusion_matrix(y_test, y_pred_rfe)}')
    
    # using chi-squared test
    model.fit(X_train_chi2, y_train)
    y_pred_chi2 = model.predict(X_test_chi2)
    acc_chi2 = accuracy_score(y_test, y_pred_chi2)
    print(f'Accuracy using chi-squared test: {acc_chi2}')
    print(f'Classification report using chi-squared test:\n{classification_report(y_test, y_pred_chi2)}')
    print(f'Confusion matrix using chi-squared test:\n{confusion_matrix(y_test, y_pred_chi2)}')
    
    print('\n')
