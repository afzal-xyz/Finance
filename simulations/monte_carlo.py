import numpy as np
from multiprocessing.pool import ThreadPool as Pool


class MonteCarloSimulation(object):

    def __init__(self, data, inputs, numberOfIterations, numberOfHistoricalDays=252, spreadDiffThreshold=5):
        """ Base MonteCarlo Simulation

        Parameters:
        data (DataFrame): Moving Average combined
        inputs (list): Number of slice
        iters (int): Number of iterations
        days (int): Number of historical days
       """
        self.data = data
        self.inputs = inputs
        self.numberOfIterations = numberOfIterations
        self.numberOfHistoricalDays = numberOfHistoricalDays
        self.spreadDiffThreshold = spreadDiffThreshold

    def annualisedSharpe(self, returns):
        """To calculate Sharpe Ratio - Sharpe ratio measures the performance of an
            investment compared to a risk-free asset, after adjusting for its risk.
            Risk free rate element excluded for simplicity

        Parameters:
        returns (DataFrame): Strategy data
        N (int): Number of historical days for simulation
       """

        return np.sqrt(self.numberOfHistoricalDays) * (returns.mean() / returns.std())


    def movingAverageStrategy(self, shortMovingAvg, longMovingAvg):
        """To calculate Moving Average Strategy

        Parameters:
        shortMovingAvg (int):
        longMovingAvg (int):
       """
        data = self.data
        """create columns with MA values"""
        data['short_ma'] = np.round(data['Close'].rolling(window=shortMovingAvg).mean(), 2)
        data['long_ma'] = np.round(data['Close'].rolling(window=longMovingAvg).mean(), 2)

        """create column with moving average spread differential"""
        data['short_ma-long_ma'] = data['short_ma'] - data['long_ma']

        """create column containing strategy 'Stance'"""
        data['Stance'] = np.where(data['short_ma-long_ma'] > self.spreadDiffThreshold, 1, 0)
        data['Stance'] = np.where(data['short_ma-long_ma'] < -self.spreadDiffThreshold, -1, data['Stance'])
        data['Stance'].value_counts()

        """create columns containing daily market log returns and strategy daily log returns"""
        data['Market Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Strategy'] = data['Market Returns'] * data['Stance'].shift(1)

        """set strategy starting equity to 1 (i.e. 100%) and generate equity curve"""
        data['Strategy Equity'] = data['Strategy'].cumsum()

        """calculate Sharpe Ratio"""
        try:
            sharpe = self.annualisedSharpe(data['Strategy'])
        except:
            sharpe = 0

        return data['Strategy'].cumsum(), sharpe, data['Strategy'].mean(), data['Strategy'].std()

    def strategy(self, inputSlices):
        """To start the simulation

        :param inputSlices:
        :return: input data, performance, sharpe ratio, mu, sigma, mc_results, mc_results_final_val
        """
        # iterate through the slice of the overall MA window tuples list that
        # has been passed to this thread

        for inputSlice in inputSlices:
            # use the current inputs to backtest the strategy and record
            # various results metrics
            perf, sharpe, mu, sigma = self.movingAverageStrategy(inputSlice[0], inputSlice[1])

            # create two empty lists to store results of MC simulation
            mc_results = []
            mc_results_final_val = []
            # run the specified number of MC simulations and store relevant results
            for j in range(self.numberOfIterations):
                daily_returns = np.random.normal(mu, sigma, self.numberOfHistoricalDays) + 1
                price_list = [1]
                for x in daily_returns:
                    price_list.append(price_list[-1] * x)

                # store the individual price path for each simulation
                mc_results.append(price_list)
                # store only the ending value of each individual price path
                mc_results_final_val.append(price_list[-1])
        return inputSlices, perf, sharpe, mu, sigma, mc_results, mc_results_final_val

def monteCarloStrategy(data, inputs, numberOfIterations, numberOfHistoricalDays):
    mc = MonteCarloSimulation(data, inputs, numberOfIterations,
                                            numberOfHistoricalDays=numberOfHistoricalDays)

class NonParallelMonteCarloSimulation():

    def __init__(self, data, inputs, numberOfIterations, numberOfHistoricalDays=252):
        """Multi-threading implementation of Monte Carlo simulation

        :param data:
        :param inputs:
        :param numberOfIterations:
        :param days:
        """
        self.inputs = inputs
        self.data = data
        self.numberOfIterations = numberOfIterations
        self.numberOfHistoricalDays = numberOfHistoricalDays

    def simulate(self):
        """Start the Parallel Monte Carlo simulations

        :return: Moving Average
        """
        pool = Pool(5)
        samples= monteCarloStrategy(self.data, self.inputs, self.numberOfIterations, self.numberOfHistoricalDays)
        return samples

class ParallelMonteCarloSimulation():

    def __init__(self, data, inputs, numberOfIterations, numberOfHistoricalDays=252):
        """Multi-threading implementation of Monte Carlo simulation

        :param data:
        :param inputs:
        :param numberOfIterations:
        :param days:
        """
        self.inputs = inputs
        self.data = data
        self.numberOfIterations = numberOfIterations
        self.numberOfHistoricalDays = numberOfHistoricalDays

    def simulate(self):
        """Start the Parallel Monte Carlo simulations

        :return: Moving Average
        """
        pool = Pool(5)
        future_res = [pool.apply_async(monteCarloStrategy,
                                       args=(self.data,
                                             self.inputs[i],
                                             self.numberOfIterations,
                                             self.numberOfHistoricalDays)) for i in range(len(self.inputs))]
        samples = [f.get() for f in future_res]
        return samples
