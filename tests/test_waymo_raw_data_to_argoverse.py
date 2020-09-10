#!/usr/bin/env python3

from waymo2argo.waymo_raw_data_to_argoverse import *

def test_round_to_micros():
    """
    test_round_to_micros()
    """
    t_nanos = 1508103378165379072
    t_micros = 1508103378165379000

    assert t_micros == round_to_micros(t_nanos, base=1000)


    