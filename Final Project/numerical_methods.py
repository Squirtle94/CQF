import pandas as pd
import numpy as np
from itertools import chain
from scipy.stats import norm
from tabulate import tabulate

def CriticalValue(level, T):
    """
    Critical values of Table 2 in the artical 'Critical Values for Cointegration Tests' from James G.MacKinnon
    Condition of:
        N=2: 2 predictors(variables)
        With constant
    Adjusted for the finite sample size: T
    """
    if level=='1%':
        cvalue = -3.89644 - 10.9519/T - 22.527/T**2
    elif level=='5%':
        cvalue = -3.33613 - 6.1101/T - 6.823/T**2
    elif level=='10%':
        cvalue = -3.04445 - 4.2412/T - 2.720/T**2
    else:
        cvalue = False
    return cvalue



def regress(X, Y):
    """
    OLS regression with:
        X: explanatory Matrix
        Y: dependent Matrix/Vector
    Return:
        a dictionary contains: coefficient, Standard error, T stats, p value, AIC, BIC
    """
    # Number of X, Y variables
    Nx = X.shape[0]
    if Y.ndim == 1:
        Ny = Y.ndim
    else:
        Ny = Y.shape[0]

    # Number of observations
    Nobs = X.shape[1]

    # calculate the coefficient
    Beta = Y @ X.T @ np.linalg.pinv((X @ X.T))

    # Calculate disturbance matrix
    Resid = Y - Beta @ X

    # Estimator of the residual covariance matrix
    Sigma = (Resid @ Resid.T) / Nobs

    # Standard error of beta coefficient
    Iinv = np.kron(np.linalg.pinv(X @ X.T), Sigma)
    std_err = np.sqrt(np.diagonal(Iinv)).reshape(Nx, Ny).T

    # T-statistics
    T = np.divide(Beta, std_err)

    # P-value
    p_value = 2 * norm.sf(np.abs(T))

    if not isinstance(Sigma, np.ndarray):
        AIC = BIC = None
    else:
        k = Nx * Ny
        # AIC
        AIC = np.log(np.linalg.det(Sigma)) + 2 * k / Nobs
        # BIC
        BIC = np.log(np.linalg.det(Sigma)) + k / Nobs * np.log(Nobs)

    res = {
        'coef': Beta,
        'std err': std_err,
        't-stats': T,
        'p-value': p_value,
        'AIC': AIC,
        'BIC': BIC,
        'resid': Resid
    }
    return res



def explanatory_matrix(X, p):
    """
    X: explanatory variable in pd.DataFrame
    p: number of lag

    return: an explanatory matrix with lag p
    """
    N = len(X)
    # Form the explanatory matrix
    if isinstance(X, pd.Series):
        Z = np.array(np.ones(N - p))
        Z = np.vstack((Z, X[p:].T))
        return Z
    else:
        Z = np.array(np.ones(N - p))
        if p == 0:
            # simply add a row of 1
            Z = np.vstack((Z, X.values.T))
        else:
            for i in range(1, p + 1):
                # form new variables
                Z = np.vstack((Z, X[p - i:N - i].values.T))
        return Z



class VAR():
    def __init__(self, Y, p):
        """
        VAR with lag p in the matrix form
        Inputs:
            Y: A DataFrame containing the variables in time series format.
            p: number of lags
        """
        self.Y = Y
        self.p = p
        # number of observations
        N = len(self.Y)
        Nobs = N - self.p
        # column names
        self.col_names = list(self.Y.columns)
        # number of variables
        Nvar = len(self.Y.columns)
        # Form the explanatory matrix
        Xmat = explanatory_matrix(self.Y, self.p)
        # Form the dependent matrix
        Ymat = self.Y[self.p:].values.T
        # OLS result
        self.res = regress(Xmat, Ymat)

        # Stability condition
        self.Eigenvalues = dict.fromkeys(range(1, self.p + 1))
        for i in range(1, self.p + 1):
            Bp = self.res['coef'][:, 1 + (i - 1) * Nvar: 1 + i * Nvar]
            eigenvalues, _ = np.linalg.eig(Bp)
            self.Eigenvalues[i] = abs(eigenvalues)

    def estimation(self):
        """
        Giving a table of Estimation, includes:
            Coefficients beta
            Standard error
            T statistics
            P value
        for the VAR of each variable
        """
        Beta = self.res['coef']
        std_err = self.res['std err']
        T = self.res['t-stats']
        p_value = self.res['p-value']

        table_header = ['Variable', 'Stats', 'Const']
        table_data = []
        for i in range(1, self.p + 1):
            table_header += [key + '(-' + str(i) + ')' for key in self.col_names]
        for i in range(len(self.col_names)):
            table_data.append([self.col_names[i], 'Estimates'] + ['{:.4f}'.format(item) for item in Beta[i]])
            table_data.append([self.col_names[i], 'Std err'] + ['{:.4f}'.format(item) for item in std_err[i]])
            table_data.append([self.col_names[i], 't-stats'] + ['{:.4f}'.format(item) for item in T[i]])
            table_data.append([self.col_names[i], 'p-value'] + ['{:.4f}'.format(item) for item in p_value[i]])
        estimation_result = pd.DataFrame(table_data, columns=table_header)
        return estimation_result

    def AIC_BIC(self, prt=True):
        """
        prt = True: Giving a table of AIC and BIC for the lag p
        prt = False: return the lag p, AIC and BIC values
        """
        AIC = self.res['AIC']
        BIC = self.res['BIC']

        if prt:
            table_header = ['Lag', 'AIC', 'BIC']
            table_data = [[str(self.p), '{:.4f}'.format(AIC), '{:.4f}'.format(BIC)]]
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        else:
            return [self.p, AIC, BIC]

    def Stability(self, prt=True):
        """
        prt = True: Giving a table of the eigenvalues of the coefficient matrix for each lag p>=1 and the corresponding stability
        prt = False: return the overall stability result for the lag p
        """
        table_header = ['p', 'Eigenvalues', 'Modulus', 'Stable']
        table_data = []
        stability = True
        for i in range(1, self.p + 1):
            for j in range(len(self.Eigenvalues[i])):
                val = self.Eigenvalues[i][j]
                if abs(val) >= 1:
                    stability = False
                if j == 0:
                    table_data.append([str(i)] + ['{:.4f}'.format(val), '{:.4f}'.format(abs(val)), abs(val) < 1])
                else:
                    table_data.append([' '] + ['{:.4f}'.format(val), '{:.4f}'.format(abs(val)), abs(val) < 1])
        if prt:
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        else:
            return stability

    def correlation(self):
        Rmat = self.res['resid']
        # get the resdiual correlation
        residual_corr = pd.DataFrame(Rmat.T, columns=self.col_names).corr()
        # return the pair correlation in descending order
        return residual_corr.stack().sort_values(ascending=False)[len(self.col_names)::2]


class EG():
    def __init__(self, X, Y):
        """
        Augmented Dickey Fuller test with lag 1 
        Inputs:
            X: independent variable in DataFrame
            Y: dependent variable in DataFrame
        """
        self.X = X
        self.Y = Y
        # number of observations
        N = len(self.X)
        # Form the explanatory matrix
        Xmat = explanatory_matrix(self.X, 0)
        # Form the dependent matrix
        Ymat = self.Y.values.T
        # OLS result
        self.res = regress(Xmat, Ymat)
        self.coef = self.res['coef']
        self.e = pd.DataFrame(self.res['resid'], index=self.X.index) 
        
    def OLS(self, prt = True):
        """
        print the estimation result of regression
        """
        table_header = ['', 'Const']
        table_data = []
        for i in range(1, len(self.coef)):
            table_header += ['beta'+str(i)]
        table_data.append(['coef'] + list(self.coef))
        table_data.append(['std err'] + list(chain.from_iterable(self.res['std err'])))
        table_data.append(['t-stats'] + list(chain.from_iterable(self.res['t-stats'])))
        table_data.append(['p-value'] + list(chain.from_iterable(self.res['p-value'])))
        if prt:
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        else:
            return self.coef
    
    def ADF_Test(self, prt=True):
        """
        print the estimation result of ADF test
        """
        # delta residual
        delta_resid = self.e.diff().dropna()
        # residual of lag 1
        resid_lag = self.e.shift(1).dropna()
        # delta residual of lag 1
        delta_resid_lag = delta_resid.shift(1).dropna()
        
        adf_data = pd.concat([resid_lag, delta_resid_lag, delta_resid], axis=1).dropna()
        adf_data.columns = ['resid_lag','delta_resid_lag', 'delta_resid']
        Nobs = len(adf_data)
        
        # explanatory matrix(with const.) and dependent matrix
        X_resid = explanatory_matrix(adf_data[['resid_lag', 'delta_resid_lag']], 0)
        Y_resid = adf_data['delta_resid'].values.T

        result = regress(X_resid, Y_resid)
        
        table_header = ['', 'Const', 'φ', 'φ1']
        table_data = []
        table_data.append(['coef'] + list(result['coef']))
        table_data.append(['std err'] + list(chain.from_iterable(result['std err'])))
        table_data.append(['t-stats'] + list(chain.from_iterable(result['t-stats'])))
        table_data.append(['p-value'] + list(chain.from_iterable(result['p-value'])))
        if prt:
            print('Critical Values:','\n 1%: ', CriticalValue('1%',Nobs), '\n 5%: ', CriticalValue('5%',Nobs), '\n 10%: ', CriticalValue('10%',Nobs))
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        else:
            return result
        
    def error_correction(self, prt = True):
        """
        This function gives the results of estimating parameters and corresponding statistics
        """
        # change of variable X
        delta_X = self.X.diff().dropna()
        # change of variable Y
        delta_Y = self.Y.diff().dropna()
        # residual of lag 1
        resid_lag = self.e.shift(1).dropna()
        # number of observed samples
        Nobs = len(delta_X)
        # form the dependent matrix
        Ymat = delta_Y.values.T
        # form the explanatory matrix (no const.)
        Xmat = pd.concat([delta_X, resid_lag], axis=1).values.T
        
        result = regress(Xmat, Ymat)
        table_header = ['', 'φ', '-(1-α)']
        table_data = []
        table_data.append(['coef'] + list(result['coef']))
        table_data.append(['std err'] + list(chain.from_iterable(result['std err'])))
        table_data.append(['t-stats'] + list(chain.from_iterable(result['t-stats'])))
        table_data.append(['p-value'] + list(chain.from_iterable(result['p-value'])))
        if prt:
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        else:
            return result

class OU():
    def __init__(self, resid):
        """
        This class gives the estimation result of fitting the residual to the Ornstein - Uhlenbeck Process
        Inputs:
            resid: the stationary residual in pd.DataFrame
        """
        self.resid = resid
        
    def fit(self, prt = True):
        """
        Fitting the residual to AR(1)
        """
        # residual of lag 1
        resid_lag = self.resid.shift(1).dropna()
        # form the dependent matrix
        dep_matrix = self.resid[1:].values.T
        # form the explanatory matrix, explanatory_matrix() function used for add a const row
        exp_matrix = explanatory_matrix(resid_lag, 0)
        # fitting
        result = regress(exp_matrix, dep_matrix)
        # coefficients
        self.C = list(result['coef'])[0]
        self.B = list(result['coef'])[1]
        # SSE
        self.SSE = result['resid']@result['resid'].T
        
        table_header = ['', 'C', 'B']
        table_data = []
        table_data.append(['coef'] + list(result['coef']))
        table_data.append(['std err'] + list(chain.from_iterable(result['std err'])))
        table_data.append(['t-stats'] + list(chain.from_iterable(result['t-stats'])))
        table_data.append(['p-value'] + list(chain.from_iterable(result['p-value'])))
        if prt:
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        
    def reversion_parameters(self, prt=True):
        """
        calculate the mean reverting parameters and return the mu_e, sigma_eq and halflife in days
        """
        # tau
        tau = 1.0/len(self.resid)
        # theta parameter
        theta = - np.log(self.B)/tau
        # mu_e parameter
        mu_e = self.C/(1-self.B)
        # sigma_eq parameter
        sigma_eq = np.sqrt(self.SSE * tau/(1-np.exp(-2*theta*tau)))
        # sigma_ou parameter
        sigma_ou = sigma_eq*np.sqrt(2*theta)
        # halflife
        tau_tilde = np.log(2)/theta
        # halflife in days
        halflife = tau_tilde/tau
        
        table_header = ['Parameters', 'Values']
        table_data = [['θ', theta],
                      ['μe', mu_e],
                      ['σeq', sigma_eq],
                      ['σou', sigma_ou],
                      ['halflife', tau_tilde],
                      ['halflife(days)', halflife]]
        if prt:
            print(tabulate(table_data, headers=table_header, tablefmt='pipe'))
        return (mu_e, sigma_eq, halflife)