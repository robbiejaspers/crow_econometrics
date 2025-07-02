# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:26:40 2025

@author: R. Jaspers
"""

import numpy as np
import pandas as pd

class autocorrelation():
    
    def autocorr(self, x: np.ndarray, lags: int) -> None:
        """Computes the autocorrelation for a given amount of lags.
        

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION.
        lags : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n: int = len(x)-lags
        x_lag = np.zeros((self.n,1+lags))
        for i in range(lags+1):
            x_lag[:,i] = x[(lags-i):(len(x)-i)]
        x_lag_mu = x_lag - x_lag.mean(axis=0)
        self.autovars: np.ndarray = (x_lag_mu.T @ x_lag_mu[:,0])/(self.n)
        self.autocorrs: np.ndarray = (self.autovars/self.autovars[0])[1:]
        
        sqrd_corrs = (self.autocorrs**2)/(self.n-(np.arange(lags)+1))
        self.ljung_box: np.ndarray = np.array([sum(sqrd_corrs[:(i+1)]) for i in range(lags)])*self.n*(self.n+2)
    
    def __str__(self):
        prt_str = ""
        for i, (c, lb) in enumerate(zip(self.autocorrs,self.ljung_box)):
            prt_str += f"p{i+1}: {round(c,3)} ({round(lb,1)})\n"
        return prt_str
        
class OLSmodel():
    
    def __init__(self):
        self.fitted = False
    
    def autocorr(self, x, lags):
        n = len(x)-lags
        x_lag = np.zeros((n,1+lags))
        for i in range(lags+1):
            x_lag[:,i] = x[(lags-i):(len(x)-i)]
        x_lag_mu = x_lag - x_lag.mean(axis=0)
        aut_var = (x_lag_mu.T @ x_lag_mu[:,0])/(n)
        return (aut_var/aut_var[0])[1:]
    
    def autocorr_summary(self, x, lags):
        n = len(x)-lags
        x_lag = np.zeros((n,1+lags))
        for i in range(lags+1):
            x_lag[:,i] = x[(lags-i):(len(x)-i)]
        x_lag_mu = x_lag - x_lag.mean(axis=0)
        aut_var = (x_lag_mu.T @ x_lag_mu[:,0])/(n)
        autocorr = (aut_var/aut_var[0])[1:]
        return (aut_var/aut_var[0])[1:]
    
    def df_fit(self,df,y_col,X_cols):
        y = df.loc[:,y_col].to_numpy()
        X = df.loc[:,X_cols].to_numpy()
        self.fit(y,X)
        self.beta_names = X_cols
    
    def fit(self, y, X):
        self.y = y
        self.X = X
        
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y # OLS beta estimate
        self.residuals = y - X @ self.beta # residuals
        self.SSR = self.residuals.T @ self.residuals # Sum of Squared Residuals
        self.error_var = self.SSR/(y.shape[0] - self.beta.shape[0]) # Error Variance
        self.SER = np.sqrt(self.error_var) # Square Root Error of Residuals
        self.R_squared = 1-self.SSR/((y.T - y.mean())@ (y - y.mean())) # R squared
        self.beta_var = self.error_var*np.linalg.inv(X.T @ X) # Variance of Beta estimates
        self.beta_SE = np.sqrt(self.beta_var.diagonal()) # Std of Beta Estimates
        self.S_hat = (self.residuals**2 * (X @ X.T)).mean() # Variance estimate
        self.S_hat_DMK = self.S_hat*len(y)/(len(y)-len(self.beta)) # Alt variance estimate
        p = X @ np.linalg.inv(X.T @ X) @ X.T
        self.S_hat_adj_d1 = (self.residuals**2 * (X @ X.T) / (1-p)).mean()
        self.S_hat_adj_d2 = (self.residuals**2 * (X @ X.T) /( (1-p)**2)).mean()
        self.beta_names = [f"b_{i}" for i in range(len(self.beta))]
        self.fitted = True
    
    def set_var_names(self,var_names):
        if len(var_names) == len(self.beta_names):
            self.beta_names = var_names
    
    def __str__(self):
        if self.fitted:
            ret_str = "|| OLS FITTED MODEL || \n"
            for n, b, e in zip(self.beta_names,self.beta,self.beta_SE):
                ret_str += f"{n}: {round(b,2)} ({round(e,2)}) \n"
            
            ret_str += f"\nSSR: {round(self.SSR,3)}, R^2: {round(self.R_squared,3)} \n"
            ret_str += f"S^: {round(self.S_hat,3)}, DMK: {round(self.S_hat_DMK,3)}, "
            ret_str += f"d1: {round(self.S_hat_adj_d1,3)}, d2: {round(self.S_hat_adj_d2,3)} \n"
        else:
            ret_str = "No model fitted yet! \n"
        return ret_str
    
    def t(self, b_n, null_b):
        self.t_stat = (self.beta[b_n] - null_b)/self.beta_SE[b_n]
    
    def F(self, R, r):
        n_r = np.linalg.matrix_rank(R)
        r_err = R @ self.beta - r
        self.F_stat = r_err.T @ np.alg.inv(R @ self.beta_var @ R.T ) @ r_err / n_r
        
    def White(self,drop_cols=None):
        mat_size = int(len(self.beta)*(len(self.beta)+1)/2)
        phi = np.zeros((len(self.y),mat_size))
        perms = [[i,j] for i in range(len(self.beta)) for j in range(i,len(self.beta))]
        for i, [j, k] in enumerate(perms):
            phi[:,i] = self.X[:,j] * self.X[:,k]
        
        if not drop_cols == None:
            for i in drop_cols:
                phi = np.delete(phi,i,1)
        
        white_y = self.residuals**2
        white_beta = np.linalg.inv(phi.T @ phi) @ phi.T @ white_y
        white_residuals = white_y - phi @ white_beta
        white_SSR = white_residuals.T @ white_residuals
        white_R_squared = 1-white_SSR/((white_y.T - white_y.mean()) @ (white_y - white_y.mean()))
        self.White_stat = len(self.y)*white_R_squared