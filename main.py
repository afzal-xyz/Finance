import numpy as np
import pandas as pd
import itertools
import time
from simulations.monte_carlo import ParallelMonteCarloSimulation


def chunk(it, size):
    """ Utility function to chunk moving average

    Parameters:
    it (list): Moving Average combined
    size (int): Number of slice

    Returns:
    int:Returning chunks

   """
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


if __name__ == "__main__":

    # read in price data
    data = pd.read_csv('F.csv', index_col='Date', parse_dates=True)

    # generate our list of possible short window length inputs
    short_mas = np.linspace(20, 50, 30, dtype=int)

    # generate our list of possible long window length inputs
    long_mas = np.linspace(100, 200, 30, dtype=int)

    # generate a list of tuples containing all combinations of
    # long and short window length possibilities
    mas_combined = list(itertools.product(short_mas, long_mas))

    # use our helper function to split the moving average tuples list
    # into slices of length 180
    mas_combined_split = list(chunk(mas_combined, 180))

    # set requried number of MC simulations per backtest optimisation
    iters = 2000

    # start timer
    start_time = time.time()

    # call our multi-threaded function
    results = ParallelMonteCarloSimulation(data, mas_combined_split, iters).run()

    # print number of seconds the process took
    print("MP--- %s seconds for para---" % (time.time() - start_time))