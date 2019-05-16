from sklearn.model_selection import cross_val_score
import numpy as np
from util.string import print_primary, print_secondary
np.random.seed(123)


def cross_validation(model_func, x_train, y_train, cv_folds, k=None,
                     results=None, scoring='explained_variance', v=1):
    print_secondary('\t scoring: %s' % scoring)
    scores = cross_val_score(
        model_func, x_train, y_train, cv=cv_folds, scoring=scoring)
    if results is not None:
        results[k] = scores
    if v:
        print('scores per fold ', [round(score, 4) for score in scores])
        print('  mean score    ', np.mean(scores))
        print('  standard dev. ', np.std(scores))


def scores_table(results):
    # render latex table
    print('Model & Mean & Std. dev. \\\\ \n\\hline')
    best_k = ''
    best_mean = 0
    for k, scores_acc in results.items():
        if np.mean(scores_acc) > best_mean:
            best_mean = np.mean(scores_acc)
            best_k = k
        print('%s & %0.4f & %0.4f\\\\' %
              (k, np.mean(scores_acc), np.std(scores_acc)))
    print_primary('\nbest score: %s, with mean: %0.4f' % (best_k, best_mean))
