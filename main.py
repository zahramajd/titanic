import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def get_titles():
    global combined
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }
    combined['Title'] = combined.Title.map(Title_Dictionary)


def process_age():
    global combined

    def fillAges(row):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    combined.Age = combined.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)


def process_names():
    global combined
    combined.drop('Name', axis=1, inplace=True)

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    combined.drop('Title', axis=1, inplace=True)


def process_fares():
    global combined
    combined.Fare.fillna(combined.Fare.mean(), inplace=True)


def process_embarked():
    global combined
    combined.Embarked.fillna('S', inplace=True)
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)


def process_cabin():
    global combined
    combined.Cabin.fillna('U', inplace=True)
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    combined.drop('Cabin', axis=1, inplace=True)


def process_sex():
    global combined
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})


def process_pclass():
    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    combined = pd.concat([combined, pclass_dummies], axis=1)
    combined.drop('Pclass', axis=1, inplace=True)


def process_ticket():
    global combined

    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = filter(lambda t: not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)


def process_family():
    global combined
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


def get_result(predicted):
    print "F1_Score: " + str(f1_score(y_validation, predicted, average='macro'))
    print "accuracy: " + str(accuracy_score(y_validation, predicted))
    print "AUC: " + str(roc_auc_score(y_validation, predicted))
    print "recall: " + str(recall_score(y_validation, predicted))
    return


# Load data
train = pd.read_csv("data_titanic/train.csv")
test = pd.read_csv("data_titanic/test.csv")
y_train = train['Survived']

combined = train.drop(['Survived'], axis=1)
combined = pd.concat([combined, test])

# Process data

# Put passengers into different category based on the titles
get_titles()

# Fill the missing age value based on the Pcalss & title
process_age()

# Drop the name col and add new cols about title that are binary
process_names()

# Fill missing fare value by the average
process_fares()

# Fill missing embarked value by the mode
process_embarked()

# binary
process_cabin()

# Replace 1 for men and 0 for women
process_sex()

# binary
process_pclass()

# binary
process_ticket()

# binary
process_family()

# Split data to validation and train
m = len(train)
n = len(test)

x_train = combined[0:m]
x_test = combined[m:m + n]

train_percent = 0.66
validate_percent = 0.33

m = len(x_train)
x_train = x_train[:int(train_percent * m)]
x_validation = x_train[int(validate_percent * m):]
y_train = y_train[:int(train_percent * m)]
y_validation = y_train[int(validate_percent * m):]

# Decision tree
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(x_train, y_train)
y_predicted_validation_dtc = dtc.predict(x_validation)
# y_predicted_test_dtc = dtc.predict(x_test)

print "- Decision tree -"
get_result(y_predicted_validation_dtc)

# Random forest
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train, y_train)
y_predicted_validation_rfc = rfc.predict(x_validation)
# y_prediction_test_rfc = rfc.predict(x_train)

print "- Random forest -"
get_result(y_predicted_validation_rfc)

# Neural network
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(x_train, y_train)
y_predicted_validation_nn = nn.predict(x_validation)
# y_predicted_test_nn = nn.predict(x_test)

print "- Neural network -"
get_result(y_predicted_validation_nn)

# Logistic regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_predicted_validation_lr = lr.predict(x_validation)
# y_prediction_test_lr = lr.predict(x_test)

print "- Logistic regression -"
get_result(y_predicted_validation_lr)

# Ensemble of classifiers

# Voting classifier
vc = VotingClassifier(estimators=[('dt', dtc), ('rf', rfc), ('lr', lr)], voting='soft')
vc.fit(x_train, y_train)
y_predicted_validation_vc = vc.predict(x_validation)
# y_prediction_test_vc = vc.predict(x_test)

print "- Voting -"
get_result(y_predicted_validation_vc)

# AdaBoost classifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
bdt.fit(x_train, y_train)
y_predicted_validation_bdt = vc.predict(x_validation)
# y_prediction_test_bdt = vc.predict(x_test)

print "- AdaBoost -"
get_result(y_predicted_validation_bdt)

# Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gb.fit(x_train, y_train)
y_predicted_validation_gb = gb.predict(x_validation)
# y_prediction_test_gb = gb.predict(x_test)

print "- Gradient Boosting -"
get_result(y_predicted_validation_gb)

# Bagging classifier
bg = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=7)
bg.fit(x_train, y_train)
y_predicted_validation_bg = bg.predict(x_validation)
# y_prediction_test_bg = bg.predict(x_test)

print "- Bagging  -"
get_result(y_predicted_validation_bg)
