def calculate_DCG(rows):
    '''
    DCG = sum of all rows(gain / log2(rang in proposal lijst))
    NDCG = (gain / log2) / iDCG
    IDCG = ideal DCG = 3/log2 1 + 3/log2 2 + 3/log2 3

    Gains:
    5 - The user purchased a room at this hotel - booking bool true
    1 - The user clicked through to see more information on this hotel - click bool true
    0 - The user neither clicked on this hotel nor purchased a room at this hotel - both click and book not true
    '''
    DCG = 0
    for row in rows.itertuples(index=True, name='Pandas'):
        click_bool = getattr(row, 'click_bool')
        position = getattr(row, 'position')
        booking_bool = getattr(row, 'booking_bool')

        if booking_bool != 1:
            if position == 1:
                DCG += click_bool / position + 5 * booking_bool / position
            else:
                DCG += click_bool / \
                    math.log2(position) + 5 * booking_bool / \
                    math.log2(position)
        else:
            if position == 1:
                # perfect score
                DCG += click_bool / position + 5 * booking_bool / position
            else:
                DCG += 5 * booking_bool / math.log2(position)
    return DCG


def rows_srch_id(data, id):
    '''
    Get all rows of a single search id
    '''
    rows = data.loc[data['srch_id'] == id]
    return rows


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


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    # https://gist.github.com/bwhite/3726239
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


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
        rows = rows_srch_id(data, id)
        r = relevance_scores(rows)
        dcg = dcg_at_k(r, r.size, method=0)
        DCG[id] = dcg
    return DCG
