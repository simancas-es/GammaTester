# GammaTester
Gamma Test class for the selection of inputs in machine learning algorithms

The goal of this class is to help by the calculations of gamma values in the Gamma Test.
Usually this test is carried out for a number of combinations and then the combination with the lowest gamma value (intercept) is chosen.
This Class stores the product of Ci-Cj for each column so that it is not calculated on every iteration.

How is it used:

1)You need:
- A dataframe with the columns of the input vectors X
- A list with the columns NAMES that are always included in the calculations, this can be []
- the Y values from dataframe[y_values].to_numpy()
- the combination of column names that are NOT FIXED columns that will be calculated eg. ['a','b','c']

2)The results are obtained from the .intercept and .slope parameters.
- The calculate function returns the deltas and gammas arrays (return self.deltas, self.gammas)

gamma_tester = GammaTester(pandas_dataframe = scaled_df,
                             fixed_columns_list = columns_always_included,
                             values_list = y,
                             p = 10)
gamma_tester.preload()
gamma_tester.calculate(column_combination = ['d','e','f'])

gamma_tester.intercept
gamma_tester.slope
gamma_tester.res2
gamma_tester.deltas
gamma_tester.gammas
gamma.terms_xeuclidean <- these are the sum of the (Ci-Cj)^2 coefficients of the euclidean distance of the last calculated combination.

Graph comparing a naive implementation without pre-caching and this version:
![Naive vs half-optimized implementation](https://github.com/simancas-es/GammaTester/blob/main/naive_vs_optimized.jpg)


To be expanded for GPUs
