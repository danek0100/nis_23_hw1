import time
import numpy as np
from arch.compat import numba


@numba.jit(nopython=True)
def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    if len(timestamps1) == 0 or len(timestamps2) == 0:
        return np.zeros(0, dtype=np.int64)

    matching = np.zeros(len(timestamps1), dtype=np.int64)

    for i, ts1 in enumerate(timestamps1):
        idx = np.searchsorted(timestamps2, ts1, side='left')
        if idx == len(timestamps2):
            matching[i] = idx - 1
        elif idx == 0 or np.abs(ts1 - timestamps2[idx-1]) > np.abs(ts1 - timestamps2[idx]):
            matching[i] = idx
        else:
            matching[i] = idx - 1

    return matching


@numba.jit(nopython=True)
def match_timestamps_two_pointers(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    if len(timestamps1) == 0 or len(timestamps2) == 0:
        return np.zeros(0, dtype=np.int64)

    '''
    ### USE THAT "Timestamps are assumed sorted" ###
    indexed_ts1 = [(t, i) for i, t in enumerate(timestamps1)]
    indexed_ts2 = [(t, i) for i, t in enumerate(timestamps2)]

    indexed_ts1.sort(key=lambda x: x[0])
    indexed_ts2.sort(key=lambda x: x[0])
    '''

    matching = np.zeros(len(timestamps1), dtype=np.int64)

    j = 0
    for i, ts1 in enumerate(timestamps1):
        while j < len(timestamps2) - 1 and abs(timestamps2[j + 1] - ts1) < abs(timestamps2[j] - ts1):
            j += 1

        matching[i] = j

    return matching


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def check_functions():
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)

    start_time_ns = time.perf_counter_ns()
    matching_two_pointers = match_timestamps_two_pointers(timestamps1, timestamps2)
    end_time_ns = time.perf_counter_ns()
    print(f"Two pointers algorithm took {end_time_ns - start_time_ns} nanoseconds")
    print(matching_two_pointers)

    start_time_ns = time.perf_counter_ns()
    matching_binary_search = match_timestamps(timestamps1, timestamps2)
    end_time_ns = time.perf_counter_ns()
    print(f"Binary search algorithm took {end_time_ns - start_time_ns} nanoseconds")
    print(matching_binary_search)

    assert np.array_equal(matching_two_pointers, matching_binary_search), "Results do not match"


def main():
    # generate timestamps for the first camera
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    # generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)
    match_timestamps(timestamps1, timestamps2)


if __name__ == '__main__':
    main()
