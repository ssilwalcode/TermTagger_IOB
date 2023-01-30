import pandas as pd
from random import seed
from sklearn.metrics import classification_report

train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

#cleaning tags
train_data['tags'] = train_data['tags'].replace("b", "B", regex=True)
train_data['tags'] = train_data['tags'].replace("II", "I", regex=True)

test_data['tags'] = test_data['tags'].replace("o", "O", regex=True)
test_data['tags'] = test_data['tags'].replace("0", "O", regex=True)

len(train_data)
train_data.head()

train_labels = train_data["tags"].to_numpy()
train_labels = [word for line in train_labels for word in line.split()]

test_sent = test_data["sent"].to_numpy()
test_sent = [word for line in test_sent for word in line.split()]

test_labels = test_data["tags"].to_numpy()
test_labels = [word for line in test_labels for word in line.split()]

# max label algorithm for classification gets the maximum label value and predicts that over the test sentences/tags --> in our case it would be 'O'
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(test))]
	return predicted

seed(1)
predictions = zero_rule_algorithm_classification(train_labels, test_sent)
print(predictions)

print(classification_report(test_labels, predictions))