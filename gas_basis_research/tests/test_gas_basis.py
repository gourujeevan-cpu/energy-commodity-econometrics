import numpy as np
import pandas as pd
import pytest
from gas_basis import _clean_prices, make_panel, walk_forward, metrics


def test_walk_forward_no_lookahead():
    rng = np.random.default_rng(9)
    dates = pd.date_range('2015-04-01', periods=110, freq='MS')
    us = 3 * np.exp(np.cumsum(rng.normal(0, .04, len(dates))))
    eu = us * np.exp(1.1 + .15 * np.sin(np.arange(len(dates))/6) + rng.normal(0, .05, len(dates)))
    p = _clean_prices(pd.DataFrame({'date': dates, 'europe_usd_mmbtu': eu, 'us_usd_mmbtu': us}))
    a, features = make_panel(p)
    pred1 = walk_forward(a, features, 48)
    p2 = p.copy()
    p2.iloc[-3, 0] *= 2
    b, features2 = make_panel(p2)
    pred2 = walk_forward(b, features2, 48)
    assert np.allclose(pred1['ridge'].to_numpy()[:-5], pred2['ridge'].to_numpy()[:-5])
    assert metrics(pred1)['out_of_sample_months'] > 40


def test_validation():
    dates = pd.date_range('2015-04-01', periods=65, freq='MS')
    rows = pd.DataFrame({'date':dates, 'europe_usd_mmbtu':5., 'us_usd_mmbtu':3.})
    rows.loc[4, 'us_usd_mmbtu'] = -1
    with pytest.raises(ValueError, match='positive'):
        _clean_prices(rows)
