import numpy as np
import util.data
import pandas as pd


def y_true(data_test: pd.DataFrame):
    # return ncdg of y_true
    y_true = data_test[['srch_id', 'click_bool', 'booking_bool']].copy()
    util.data.add_score(y_true)
    util.data.add_position(y_true)
    return mean(y_true)


def y_pred(x_test: pd.DataFrame, y_pred: np.ndarray):
    # return ncdg of the model prediction `y_pred`
    y = util.data.to_df(x_test, y_pred)
    util.data.add_position(y)
    return mean(y)


def mean(data):
    return np.mean(list(util.ndcg.DCG_dict(data).values()))


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    # https://gist.github.com/bwhite/3726239
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def relevance_scores(rows):
    positions = rows['position']
    p_max = int(positions.max()) + 1
    r = np.zeros(p_max)
    position = 1
    for row in rows.itertuples(index=True, name='Pandas'):
        # click_bool = int(getattr(row, 'click_bool'))
        # position = int(getattr(row, 'position'))
        # booking_bool = getattr(row, 'booking_bool')
        score = getattr(row, 'score')
        r[position] = score
        position += 1
        # if booking_bool > 0:
        #     r[position] = 5
        # else:
        #     r[position] = 1 * click_bool
    return r


def DCG_dict(data):
    DCG = {}
    for id in data['srch_id'].unique():
        # rows = rows_srch_id(data, id)
        # r = relevance_scores(rows)
        r = []
        prev_srch_id = -1
        position = 1
        for i in data.index.tolist():
            score = data.score
            row = data.loc[i]
            # compute position
            if prev_srch_id != row.srch_id:
                prev_srch_id = row.srch_id
                break
            else:
                r.append(score)
                position += 1
        dcg = dcg_at_k(r, len(r), method=0)
        DCG[id] = dcg
    return DCG
