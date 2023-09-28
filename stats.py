import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def calc_stats(true, pred):
    """
    :param true: true labels
    :param pred: predicted labels
    :return: cm - confusion matrix, prec - precision per class, rec - recall per class, f_score - f score per class
    """
    true = np.array(true)
    true = true.flatten()
    pred = pred.flatten()

    cm = confusion_matrix(true,
                          pred,
                          labels=[0., 1., 2., 3., 4.])

    prec = {'soil': cm[0, 0] / np.sum(cm[:, 0]),
            'bedrock': cm[1, 1] / np.sum(cm[:, 1]),
            'sand': cm[2, 2] / np.sum(cm[:, 2]),
            'big rock': cm[3, 3] / np.sum(cm[:, 3]),
            'null': cm[3, 3] / np.sum(cm[:, 3])
            }
    rec = {'soil': cm[0, 0] / np.sum(cm[0, :]),
           'bedrock': cm[1, 1] / np.sum(cm[1, :]),
           'sand': cm[2, 2] / np.sum(cm[2, :]),
           'big rock': cm[3, 3] / np.sum(cm[3, :]),
           'null': cm[4, 4] / np.sum(cm[3, :])
           }
    f_score = {'soil': 2 * prec['soil'] * rec['soil'] / (prec['soil'] + rec['soil']),
               'bedrock': 2 * prec['bedrock'] * rec['bedrock'] / (prec['bedrock'] + rec['bedrock']),
               'sand': 2 * prec['sand'] * rec['sand'] / (prec['sand'] + rec['sand']),
               'big rock': 2 * prec['big rock'] * rec['big rock'] / (prec['big rock'] + rec['big rock']),
               'null': 2 * prec['null'] * rec['null'] / (prec['null'] + rec['null'])
               }

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[0., 1., 2., 3., 4.])
    disp.plot(cmap='Blues')
    plt.show()

    return cm, prec, rec, f_score
