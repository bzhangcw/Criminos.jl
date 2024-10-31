import numpy as np


def create_rec(row, T):
    array = np.zeros(T, dtype=int)
    _T = np.ceil(row["Np"] / row["p"]).astype(int)
    slots = np.random.choice(np.arange(1, _T + 1), row["Np"], replace=False)
    array[T - 1 - slots] = 1
    array[-1] = row["Recidivism_Arrest_Year1"]
    return array


def create_rec_exp(row, T):
    array = np.zeros(T, dtype=int)
    # total number of arrivals
    _T = np.ceil(row["Np"] / row["p"]).astype(int)
    # compute μ: according to row["p"]
    mu = -np.log(1 - row["p"])
    current_time = 0.0
    array[-1] = row["Recidivism_Arrest_Year1"]
    arrivals = 0
    while True:
        # Generate an exponential interarrival time
        interarrival_time = np.random.exponential(scale=1 / mu)
        current_time += interarrival_time
        if current_time > T or arrivals >= row["Np"]:
            break
        array[-int(current_time)] = 1
        arrivals += 1
    return array


def create_rec_simple(row, T):
    array = np.zeros(T, dtype=int)
    # total number of arrivals
    _T = np.ceil(row["Np"] / row["p"]).astype(int)
    # compute μ: according to row["p"]
    mu = -np.log(1 - row["p"])
    current_time = 0.0
    array[-1] = row["Recidivism_Arrest_Year1"]
    arrivals = 0
    for j in range(1, T):
        # Generate an exponential interarrival time
        zj = np.random.rand() < row["p"]
        array[j] = zj
        arrivals += zj
        if arrivals >= row["Np"]:
            break
    return array
