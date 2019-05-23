import numpy as np
import util.data


def ndcg(X_test, y_test, y_pred, ):
    Xy_pred = X_test.copy([['srch_id', 'prop_id', 'score']])
    Xy_pred['score_pred'] = y_pred
    Xy_pred['score'] = y_test
    Xy_pred.sort_values(['srch_id', 'score_pred'], ascending=[True, False])
    dcg_test = DCG_dict(Xy_pred)
    ndcg = np.mean(np.array(list(dcg_test.values())))
    return ndcg


def sort_pred_test(x_test, y_test, y_pred):
    # calculate dcg of test set per srch_id
    Xy_pred = util.data.Xy_pred(x_test, y_pred)
    # put true y values on indexes, do not sort !
    Xy_true = util.data.Xy_pred(x_test, y_test)
    return Xy_pred, Xy_true


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def DCG_dict(data):
    DCG = {}
#     for id in data['srch_id']:
    # rows = rows_srch_id(data, id)
    # r = relevance_scores(rows)
    r = []
    prev_srch_id = -1
    position = 0
    for i in data.index.tolist():
        if prev_srch_id == -1:
            row = data.loc[i]
            cur_srch_id = row.srch_id
            prev_srch_id = 0
        row = data.loc[i]
        next_id = row.srch_id
        score = row.score
        # compute position
        if cur_srch_id != next_id:
            DCG[cur_srch_id] = ndcg_at_k(r, k=len(r))
            cur_srch_id = next_id
            r = []
            r.append(score)
            position += 1
        else:
            r.append(score)
            position += 1
    DCG[cur_srch_id] = ndcg_at_k(r, k=len(r))
    return DCG
