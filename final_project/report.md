# Intro to Machine Learning Final Project
### Identify Fraud from Enron Email

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

__Project background and goals__

This project involves programmatically identifying fraudulent Enron employees using financial data as well as data extracted from
the Enron email corpus. Enron was a U.S. energy-trading company whose executives hid debt and exaggerrated growth through accounting
loopholes. At its peak, Enron was placed sixth on the _Fortune_ Global 500. The uncovering of Enron's fraudulent practices and
ensuing scandal resulted in the then-largest bankruptcy reorganization in American history and the sentencing of many of the company's
executives. The dataset consists of financial information as well as a corpus of about 500,000 emails for about 140 key Enron
employees.

<<ML STUFF>>

__Outlier handling__

Although a large amount of variation was expected for the financial data, sorting the data revealed two non-employee entires in the
data dictionary: 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. These entries were removed from the dataset.

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

All of the given features were used, plus three additionally engineered features: `fraction_from_poi`, `fraction_to_poi`, and
`fraction_restricted_stock`. These features were the proportion of emails received from persons of interest, sent to persons of
interest, and restricted stock holdings, respectivly. The rationale behind engineering these features was the potential to gain more
information by placing the component features in context as ratios. The usage of all features was decided upon after trying several
combinations, including solely financial or email features.

Due to the usage of a decision tree classifier, no feature scaling was required. The most important features in the final decision tree
were `bonus` (24.6%) and `fraction_restricted_stock` (19.3%).

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

The final choice of classifier was a decision tree, with a final precision of 0.311 and recall of 0.336. A random forest algorithm
was also applied to the problem, but precision and recall both suffered by 0.05 to 0.07 when using the most optimized parameters.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.