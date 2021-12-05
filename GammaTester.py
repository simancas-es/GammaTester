# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:14:22 2021
GAMMA TEST FOR INPUT SELECTION

@author: Jose Antonio
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from time import perf_counter

class GammaTester:

    def __init__(self,pandas_dataframe = None, fixed_columns_list = None,
                 row_list=None,values_list=None,
                 p=10):
        self.p=p
        
        self.pandas_dataframe = pandas_dataframe
        self.fixed_columns_list = fixed_columns_list
        self.values_list=np.array(values_list)
        
        self.num_rows = None
        self.basis_euclidean = None
        self.precalculated_pairs = None
        self.LOADED = False
                
        self.basis_euclidean = None
        self.euclidean_matrix=None
        self.deltas= []
        self.gammas= []
        
        if (pandas_dataframe is not None and
            fixed_columns_list is not None and
            values_list is not None):
            self.preload()

        
        self.intercept=None
        self.slope=None
    
    def preload(self):
        #We calculate the unmuted Col[i] - C[j]

        unchanged_df = self.pandas_dataframe[self.fixed_columns_list]
        df = self.pandas_dataframe       
        
        self.num_rows = len(df)  
        self.basis_euclidean = np.zeros((self.num_rows,self.num_rows))
        for col in unchanged_df:
            A = unchanged_df[col].to_numpy()
            #A2[0] is the column minus the element from the first row
            #A2[0][1] is the squared result of the first row minus the second row
            self.basis_euclidean += np.array([A-x for x in A])**2
            #TODO: Improve symmetric matrix
        
        changing_cols = [col for col in df.columns
                         if col not in self.fixed_columns_list]
        #Same saved-up combinations of Ci - Cj ^2 for the rest of the columns(not summed up yet)         
        self.precalculated_pairs = dict(zip(changing_cols,
                                            [np.triu(np.array([df[col].to_numpy()-x for x in df[col].to_numpy()])**2)
                                                     for col in df]
                                            )
                                        )
                                        
        self.LOADED = True
        
    def has_collision(self,column_combination):
        for col in column_combination:
            # error_columns = []
            if col in self.fixed_columns_list:
                # error_columns.append(col)
                return True                
        return False
    
    def calculate(self, column_combination = None,
                            p= None, check_collision = True):
        
        if not self.LOADED:
            raise ValueError('GammaTester not preloaded. Use preload() or reinitialize.')
        if check_collision:
            if self.has_collision(column_combination):
                raise ValueError('Some calculated columns coincide with the already calculated ones.')
        if p is None: p=self.p

        df = self.pandas_dataframe        
        num_rows = self.num_rows
        
        # =============================================================================
        #     E U C L I D E A N   D I S T A N C E S
        # =============================================================================
        
        
        #We calculate the sum of all the (Ci - Cj)**2 and the fixed rows
        #Then square the result and divide it by the number of rows
        euclidean_distances = np.triu(self.basis_euclidean.copy())
        for col in column_combination:
            euclidean_distances+=self.precalculated_pairs[col]
        euclidean_distances = euclidean_distances**0.5    
        # print(f'euclidean_value (i,j) :\n {euclidean_distances}')
        
        #for each row, sort and get the indices of the sorting
        euclidean_distances_symmetric = euclidean_distances + euclidean_distances.T
        indices_order = np.argsort(euclidean_distances_symmetric)
        euclidean_distances_symmetric.sort()
        # print(f'euclidean symmetric sorted :\n {euclidean_distances_symmetric}')
        self.deltas = (euclidean_distances_symmetric[:,1:p+1]**2).sum(axis=0)/num_rows
        # print(f'deltas :\n {deltas}')
        self.deltas = self.deltas.reshape(-1,1)

        
        y = self.values_list
        # print(f'y: \n{y} \n y ordered {y[indices_order]}')
        self.gammas = ((y[indices_order][:,1:p+1].T-y)**2).sum(1)/num_rows/2
        
        model = LinearRegression().fit(self.deltas,self.gammas)
        self.intercept=model.intercept_
        self.slope=model.coef_
                       
        return self.deltas, self.gammas



if __name__=="__main__":
 
    # =============================================================================
    #     P E R F O R M A N C E 1
    # =============================================================================
    # import matplotlib.pyplot as plt
    # NUM_TRIES = 100
    # def inventarse_data(nrows,ncolumns):
    #     columns = [str(i) for i in range(ncolumns)]
    #     data = np.random.random_sample((nrows, ncolumns)) - 2
    #     values_list = np.sin(data.sum(axis=1))
    #     return columns,data,values_list
    
    
    # columns, row_list, values_list = inventarse_data(100,NUM_TRIES)
    # df = pd.DataFrame(data=row_list, columns = columns)
    
    # fixed_columns = []
    # combinations = [str(w) for w in range(0,NUM_TRIES)]   

    # t1_start = perf_counter()
    # time_elapsed_2=[]
    # gt2 = GammaTester(pandas_dataframe = df,
    #                   fixed_columns_list = fixed_columns,
    #                   values_list = values_list,
    #                   p=3)
    
    # for i in range(0,NUM_TRIES):
    #     gt2.calculate(column_combination=combinations[0:i])
    #     time_elapsed_2.append(perf_counter()-t1_start)

    # plt.scatter(x=range(0,NUM_TRIES),y = time_elapsed_2,label = "Optimizado")
    # plt.legend()
    # plt.ylabel("ms")
    # plt.xlabel("numero de columnas")
    pass
    