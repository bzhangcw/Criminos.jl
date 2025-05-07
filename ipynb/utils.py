from header import *


def sum_y_and_count_groupby_N(y, cumsum):
    M, N = y.shape
    results = []

    for t in range(N):
        # Create a DataFrame for easier grouping
        df = pd.DataFrame({"y": y[:, t], "cumsum": cumsum[:, t]})

        # Group by the cumulative sum
        grouped_sum = df.groupby("cumsum")["y"].sum()
        grouped_count = df.groupby("cumsum")["y"].count()

        # Combine the sum and count into one DataFrame
        result = pd.DataFrame({"$y$": grouped_sum, "$x$": grouped_count})
        results.append(result)

    return results


def find_first(arr):
    indices = np.where(arr == 1)[0]
    index = indices[0] if indices.size > 0 else -1
    return index


# import all functions from records.py
from records import *

# import all functions from plots.py
from plots import *
