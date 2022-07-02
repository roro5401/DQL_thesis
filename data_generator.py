import random
import numpy as np
import pandas
import pandas as pd
import math

"""
This function creates a dataset, pandas DataFrame, according to a Weibull distribution.

@param alpha: scale parameter for the Weibull Distribution.
@param beta: shape parameter for the Weibull Distribution.
@n_turbines: number of wind turbines, size of the wind park.
@n_periods: maximum number of periods in one simulation.
@seed: random seet for the numpy package.
"""
def create_dataset(alpha, beta, n_turbines, n_periods = 180, seed = 1):
    np.random.seed(seed)
    life_times = alpha*np.random.weibull(beta, (n_turbines, n_periods))
    df = pd.DataFrame(columns=['id', 'age', 'next failure', 'lifetimes'])

    # setting up the data frame
    for i in range(0, len(life_times)):
        max_age = math.ceil(life_times[i, 0])
        if max_age == 1: max_age = 2
        random.seed(seed)
        current_age = random.randrange(1, max_age)
        next_failure = max_age - current_age
        d = pandas.DataFrame([{'id': i, 'age': current_age, 'next_failure': next_failure, 'lifetimes': np.ceil(life_times[i, 1:])}])
        df = pd.concat([df, d], axis=0, ignore_index=True)
    return df
