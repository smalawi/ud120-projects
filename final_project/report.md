# Intro to Machine Learning Final Project
### Identify Fraud from Enron Email

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

__Project background and goals__

This project involves programmatically identifying fraudulent Enron employees using financial data as well as data extracted from
the Enron email corpus. Enron was a U.S. energy-trading company whose executives hid debt and exaggerrated growth through accounting
loopholes. At its peak, Enron was placed sixth on the _Fortune_ Global 500. The uncovering of Enron's fraudulent practices and
ensuing scandal resulted in the then-largest bankruptcy reorganization in American history and the sentencing of many of the company's
executives.

The dataset consists of financial information as well as a corpus of about 500,000 emails for 146 key Enron
employees, 18 of which are persons of interest (POI). A total of 21 features are provided for initial investigation.
Using this data, a machine learning algorithm can be built to predict whether a given employee is a POI.

__Outlier and missing data handling__

Although a large amount of variation was expected for the financial data, sorting the data revealed two non-employee entires in the
data dictionary: 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. These entries were removed from the dataset.

A number of employees had missing data for various fields. Missing email information was imputed to the overall mean for the given
field, while missing financial information was imputed to the median rather than the mean, since executives were expected to exhibit
outliers in some financial fields (which would skew the average).

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

All of the given features were used. Three additional unused features were engineered and tested: `fraction_from_poi`,
`fraction_to_poi`, and
`fraction_restricted_stock`. These features were the proportion of emails received from persons of interest, sent to persons of
interest, and restricted stock holdings, respectivly. The rationale behind engineering these features was the potential to gain more
information by placing the component features in context as ratios.

The usage of all features was decided upon after trying several combinations, including solely financial or email features.
Classifier performance decreased slightly when using the new features, so they were left unused in the final classifier. Performance
suffered considerably when only using a subset of the features, as displayed in the table below:

| Featureset     | New features included? | Precision | Recall | Accuracy |
| -------------- | ---------------------  | --------- | ------ | -------- |
| Financial      | No                     | 0.248     | 0.410  | 0.756    |
| Email          | No                     | 0.267     | 0.521  | 0.745    |
| Combined       | No                     | 0.330     | 0.555  | 0.791    |
| Combined       | Yes                    | 0.308     | 0.470  | 0.789    |

Due to the usage of a random forest classifier, no feature scaling was required. The most important features in the final random
forest were `bonus` (35.1%) and `restricted_stock` (21.3%).

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

The final choice of classifier was a random forest. A decision tree algorithm
was also applied to the problem, but recall suffered significantly when using the most optimized parameters.
The final performances as measured by precision and recall are reported below:

| Algorithm      | Precision | Recall |
| -------------- | --------- | ------ |
| Decision Tree  | 0.311     | 0.336  |
| Random Forest  | 0.308     | 0.470  |

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?

Parameter tuning refers to the adjustment of a machine learning algorithm's parameters to achieve optimal performance. Adjusting
parameters can have just as significant an effect on classifier performance as selecting an entirely different algorithm. A badly 
tuned algorithm will be significantly less useful for making predictions.

Several tools exist for automatic the parameter tuning process. For this project, grid search cross validation was used to quickly
find the optimal combination of parameters from a supplied matrix of possibilities. The final random forest classifier had the
following parameters:

* 'n_estimators'           : [10, 20, __30__]
* 'criterion'              : ['gini', __'entropy'__]
* 'bootstrap'              : [True, __False__]
* 'max_depth'              : [__2__, 10, 20]
* 'min\\_samples\\_split'  : [__2__, 4, 6, 8]
* 'class_weight'           : [None, __'balanced'__]

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation entails testing a machine learning algorithm's performance after the training process. The purpose of validation is to
ensure that an algorithm is trained such that it can be applied to additional data without sacrificing performance. Accordingly, a
common mistake is to validate the algorithm on the same set of data it was trained on, which defeats the purpose of checking the
algorithm's performance on multiple sets of data.

The algorithm used in this project was validated by uniformly randomly splitting the given data into
training and testing sets (70% and 30%, respectively). Due to the class imbalance in this problem, the uage of a stratified shuffle
split (to even out the number of POIs and non-POIs in training and testing data) was experimented with, but final classifier 
performance decreased slightly from a normal train-test split.

### Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

The metrics by which the algorithm was evaluated were precision and recall. These refer to the proportion of positive _results_ that
were correctly classified (true positives), and the proportion of positive _data entries_ succesfully retrieved, respectively. In
other words, a high precision indicates a low proportion of false positives, while a high recall indicates a low proportion of false
negatives.

For this project, precision and recall are more useful metrics than accuracy because they give more insight in the context of this
problem's class imbalance; the persons of interest in the dataset are a distinct minority. Since a classifier could still attain a
decent accuracy by simply identifying all data points as negative, precision and recall are better choices for understanding a 
classifier's performance here. For this problem, precision is equivalent to the proportion of employees classified as POIs who are
actually POIs, and recall is equivalent to the proportion of POIs who are successfully classified. One might argue that a higher
precision is desirable in this case; if the algorithm is used as an initial automated screening tool, it would be more important
to not miss any fraudulent employees than to avoid misclassifying innocent ones.

As reported above, the final random forest classifier exhibited a precision of 0.308 and recall of 0.470.