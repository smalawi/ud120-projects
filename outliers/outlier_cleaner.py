#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    with_errors = []
    for pred, age, act in zip(predictions, ages, net_worths):
        with_errors.append((age[0], act[0], (pred[0] - act[0])**2))
    sorted_data = sorted(with_errors, key=lambda t: t[2], reverse=True)

    for i in range(len(sorted_data) / 10, len(sorted_data)):
        cleaned_data.append(sorted_data[i])

    print len(cleaned_data)
    
    return cleaned_data

