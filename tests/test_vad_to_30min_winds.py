import datetime as dt
import pytz
from vad_to_30min_winds import create_time_ranges


def test_create_time_ranges():
    day = dt.date(2021, 6, 30)
    res = create_time_ranges(day)
    assert len(res) == 48
    assert res[0] == dt.datetime(2021, 6, 30, 00, 00, 00, 00, pytz.UTC)
    assert res[-1] == dt.datetime(2021, 6, 30, 23, 30, 00, 00, pytz.UTC)
