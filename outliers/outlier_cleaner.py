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

    # calculate R squared for each point
    individual_errors = (net_worths - predictions)**2

    # create a triple with the error value
    for i in range(len(predictions)):
        cleaned_data += [(ages[i][0], net_worths[i][0],
                          individual_errors[i][0])]

    # sort the data based on the error rate
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2], reverse=True)

    # we want to remove 10% of points
    length = int(len(predictions) * 0.1)

    return cleaned_data[length:]
