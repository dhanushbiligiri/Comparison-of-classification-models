# Comparison-of-classification-models
Project overview: The objective of this project is to evaluate various feature selection strategies and spam
email classification systems. We want to look into how well feature selection methods like
PCA, Recursive Feature Elimination performs and Chi Squared test. We also want to look into how well various
classification methods like Naive Bayes, Random Forest, AdaBoost, Decision Trees, K-nearest
neighbors, and Support Vector Machines perform. The goal is to identify the feature
selection and classification technique combination that produces the most accurate
classification of spam.

Data Resources:
The csv file contains 5172 rows, each row for each email. There are 3002 columns. The first column indicates Email name. The name has been set with numbers and not recipients' name to protect privacy. The last column has the labels for prediction : 1 for spam, 0 for not spam. The remaining 3000 columns are the 3000 most common words in all the emails, after excluding the non-alphabetical characters/words. For each row, the count of each word(column) in that email(row) is stored in the respective cells. Thus, information regarding all 5172 emails are stored in a compact dataframe rather than as separate text files.
Link - https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv


Output:





1. Random forest:





![image](https://user-images.githubusercontent.com/62478652/224862182-854e4c0f-2057-43d8-86f2-1728ff63dc3a.png)





2.AdaBoost





![image](https://user-images.githubusercontent.com/62478652/224862262-449ab136-374e-49db-b7e6-c40b82cdf5d4.png)





3.Decision Tree










![image](https://user-images.githubusercontent.com/62478652/224862332-43982d0a-e9e3-4383-8c9a-e48eb613eab2.png)





4.KNN





![image](https://user-images.githubusercontent.com/62478652/224862376-b03f4534-58d8-4394-abb9-588e0c719cbe.png)





5.SVM





![image](https://user-images.githubusercontent.com/62478652/224862432-41af0808-97ac-472a-8413-6f2164d2ebf8.png)
