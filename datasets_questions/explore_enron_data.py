#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

poi = [x for x in enron_data if enron_data[x]["poi"] == 1]
# print len(poi)

# print enron_data["LAY KENNETH L"]["total_payments"]

people = [x for x in enron_data if enron_data[x]["total_payments"] == 'NaN']
print len(people)
print len(enron_data)