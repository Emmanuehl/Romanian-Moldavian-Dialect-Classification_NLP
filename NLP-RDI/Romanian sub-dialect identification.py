
# import zone

import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import MultinomialNB


# Natural Language Processing

file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/train_samples.txt",
            encoding="utf8")
train1_samples = file.readlines()

file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/validation_samples.txt",
            encoding="utf8")
train2_samples = file.readlines()

samples1 = []
for x in train1_samples:
    samples1.append(x.split("\t")[1])
    
samples2 = []
for y in train2_samples:
    samples2.append(y.split("\t")[1])

file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/train_labels.txt",
            encoding="utf8")
train1_labels = file.readlines()
file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/validation_labels.txt",
            encoding="utf8")
train2_labels = file.readlines()

labels1 = []
for x in train1_labels:
    labels1.append(x.split()[1])    
labels2 = []
for y in train2_labels:
    labels2.append(y.split()[1])

data_samples = []
data_samples = samples1 + samples2
data_labels = []
data_labels = labels1 + labels2


# Feature extraction
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(data_samples)

# Classifier
model = MultinomialNB()
model.fit(features, data_labels)

file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/validation_samples.txt",
            encoding="utf8")
test_samples = file.readlines()
file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/validation_labels.txt",
            encoding="utf8")
test_labels = file.readlines()

testing_samples = []
for x in test_samples:
    testing_samples.append(x.split("\t")[1])
    
testing_labels = []
for y in test_labels:
    testing_labels.append(y.split()[1])

test_features = vectorizer.transform(testing_samples)
prediction = model.predict(test_features)

# Evaluation of accuracy and precision
f1_score(testing_labels, prediction, average='macro')

print('Confusion Matrix :')
results = confusion_matrix(testing_labels, prediction)
print(results)
print('Accuracy Score :', accuracy_score(testing_labels, prediction))
print('Report : ')
print(classification_report(testing_labels, prediction))


file = open("C:/Users/Emanuel/Desktop/romanian-sub-dialect-identification-master/test_samples.txt",
            encoding="utf8")
samples = file.readlines()

submit_samples = []
id = []
for x in samples:
    submit_samples.append(x.split('\t')[1])
    id.append(x.split()[0])

submit_features = vectorizer.transform(submit_samples)
submit_prediction = model.predict(submit_features)

# Writing to CSV

with open('emi.csv', 'w', newline='') as f:
    filewriter = csv.writer(f)
    filewriter.writerow(['id', 'label'])
    for i in range(len(submit_prediction)):
        filewriter.writerow([id[i], submit_prediction[i]])
