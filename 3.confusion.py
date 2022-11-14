from sklearn import metrics

actual = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
predicted = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0]

tm, fm = metrics.confusion_matrix(actual, predicted)
tn = tm[0]
fn = tm[1]
fp = fm[0]
tp = fm[1]

accuracy = (tp + tn)/(tp + tn + fp + fn)
sensitivity = tp/(tn + fn)
sensitivity = "{:2f}".format(sensitivity)
precision = tp/(tp + fp)
precision = "{:2f}".format(precision)
specificity = tn/(tn + fp)

print({"Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity})