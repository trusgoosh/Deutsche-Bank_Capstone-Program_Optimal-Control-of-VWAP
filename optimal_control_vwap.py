#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: william_chen
"""


import pandas
import matplotlib.pyplot as plt
import numpy
from scipy import integrate
import math


# generate a discrete gamma bridge, with partition density specified by length parameter
def gammabridge(m, v, length):
    k = m ** 2 / v
    theta = v / m
    gamma_gap = numpy.random.gamma(k, theta, length)
    cummulative_gamma = [0]+[sum(gamma_gap[:i+1]) for i in range(length)]
    bridge = [cummulative_gamma[i]/cummulative_gamma[-1] for i in range(len(cummulative_gamma))]
    return bridge


# function a in Theorem 3.1
def a(t, kai=1e-8, lambda_=1, sigma=0.01, T=1):
    A = math.sqrt(kai*lambda_*sigma**2)
    B = math.exp(2*T*math.sqrt(lambda_*sigma**2/kai))
    C = math.exp(2*t*math.sqrt(lambda_*sigma**2/kai))
    return A * (B + C) / (B - C)


# function b in Theorem 3.1
def b(t, kai=1e-8, T=1):
    return -2 * a(t, kai=kai, T=T) - c(t, kai=kai, T=T)


# function c in Theorem 3.1
def c(t, kai=1e-8, T=1):
    return -2 * kai / (T - t)
    

# generate a left-continuous step function from a discrete gamma bridge, for integration purpose
def gamma(t, gamma_bridge, x1=0, x2=1):
    if t < x1 or t > x2:
        return
    else:
        delta = (x2 - x1) / (len(gamma_bridge) - 1)
        # gamma is left continuous
        which = int((t - x1) / delta)
        return gamma_bridge[which]


# represents the most inner integrand in the optimal share holding function
def function1(t, b, gamma, gamma_bridge, c, a, s, kai=1e-8):
    return (b(t, kai=kai) * gamma(t, gamma_bridge) + c(t, kai=kai)) * math.exp(-(1/kai)*integrate.quad(lambda x: a(x, kai=kai), t, s)[0])


# share holding trajectory with respect to optimal control u, (t, x0) is the initial condition
def x(s, gamma_bridge, t=0, x0=0, kai=1e-8):
    if s < t:
        return
    else:
        A = x0 * math.exp(-1/kai*integrate.quad(a, t, s)[0])
        B = -(1/(2*kai))*integrate.quad(lambda y: function1(y, b, gamma, gamma_bridge, c, a, s, kai), t, s)[0]
        return A + B


# function descriptions
# turns tick data into 5-minute data (price changes and trading volume)
# calculates correlation between relative volume and total trading volume
# calculates correlation between trading volume and price changes
# input: tick-level trading data
# returns: 1. trading volume in 5-minute intervals
#           2. price changes for 5-minute intervals
#           3. correlation series between relative volume and total trading volume
#           4. correlation series between trading volume and price changes
def corranal(data):
    
    # list of dates for the data
    dates = numpy.sort(list(set(data['date'])))
    
    # generate 5-minute intervals in strings
    t = []
    for hour in ['09', '10', '11', '13', '14', '15']:
        for minute in range(0, 56, 5):
            if minute < 10:
                min_ = '0' + str(minute)
            else:
                min_ = str(minute)
            t.append(hour + ':' + min_)
    t = t[6:] + ['16:00']
    
    df = pandas.DataFrame(index=range(66), columns=dates)# df contains trading volume in each 5-min interval
    df_pc = pandas.DataFrame(index=range(66), columns=dates)# dataframe of price changes
    
    for i in range(len(dates)):
        #slice out data in a specific date
        temp = data[data['date']==dates[i]]
        print(dates[i])
        for j in range(len(t)-1):
            temp1 = temp[temp['time']>=t[j]]
            temp1 = temp1[temp1['time']<t[j+1]]
            temp1 = temp1.set_index(pandas.Index(range(len(temp1))))
            df.iloc[j, i] = temp1['size'].sum()
            if len(temp1) != 0:
                df_pc.iloc[j, i] = temp1['price'][len(temp1)-1] - temp1['price'][0]
            else:
                df_pc.iloc[j, i] = 0.0
    
    df_accumulativeVolume = df.copy()# cumulative trading volumes
    df_relativeVolume = df.copy()# relative trading volumes
    for i in range(len(dates)):
        for j in range(len(df)-1):
            df_accumulativeVolume.iloc[j+1, i] = df_accumulativeVolume.iloc[j+1, i] + df_accumulativeVolume.iloc[j, i]
    for i in range(len(dates)):
        for j in range(len(df)):
            df_relativeVolume.iloc[j, i] = df_accumulativeVolume.iloc[j, i]/df_accumulativeVolume.iloc[-1, i]
    
    # correlation between relative volume and total trading volume
    corr1 = [0 for i in range(len(df)-1)]
    for i in range(len(df)-1):
        corr1[i] = numpy.corrcoef(df_relativeVolume.iloc[i, :], df_accumulativeVolume.iloc[-1, :])[0, 1]
    
    # correlation between trading volume and price changes
    corr2 = [0 for i in range(len(dates))]
    for i in range(len(dates)):
        corr2[i] = numpy.corrcoef(df.iloc[:, i].tolist(), df_pc.iloc[:, i].tolist())[0, 1]
    
    return df, df_pc, corr1, corr2


if __name__ == '__main__':
    
    stock_file_lst = ["aia.csv", "hsbc.csv", "tencent.csv", "pingan.csv"]
    result_dict = {}
    
    # calculate 5-minute trade data and correlation data, stored in dictionary
    for i in range(len(stock_file_lst)):
        data_path = stock_file_lst[i]
        data = pandas.read_csv(data_path)
        data = data[pandas.isnull(data['cond'])]# excludes trades with bad conditions
        data = data.set_index(pandas.Index(range(len(data))))
        data['time'] = data['time'].str.slice(0, 5)
        result_dict[stock_file_lst[i][:-4]] = corranal(data)
    
    # 66 5-minute intervals in a trading day, plotting purpose
    time = numpy.linspace(0, 1, 67)
    # list of dates for the data, plotting purpose
    dates = numpy.sort(list(set(data['date'])))
    
    # mean and variance for simulation parameter estimation
    m = result_dict["hsbc"][0].mean().mean()
    var = result_dict["hsbc"][0].var().mean()
    bridge = gammabridge(m, var, 66)
    
    # plot a gamma bridge
    plt.step(time, bridge, where='post')
    plt.xlabel('time')
    plt.ylabel('relative volume')
    plt.show()
    
    #plot relative trading volumes
    for i in range(2):
        df = result_dict[stock_file_lst[i][:-4]][0]
        df1 = df.copy()# deep copy dataframe
        # calculate relative trading volume
        for j in range(len(dates)):
            for k in range(len(df1)-1):
                df1.iloc[k+1, j] = df1.iloc[k+1, j] + df1.iloc[k, j]
            df1[dates[j]] = df1[dates[j]] / df1[dates[j]][len(df1)-1]
        plt.plot(df1)
        plt.xlabel('time in a trading day')
        plt.ylabel('relative volume')
        plt.title(stock_file_lst[i][:-4])
        plt.show()
    
    # plot gamma bridge simulations using parameters from certain stocks 
    for i in range(2):
        m = result_dict[stock_file_lst[i][:-4]][0].mean().mean()
        var = result_dict[stock_file_lst[i][:-4]][0].var().mean()
        plt.figure()
        for j in range(len(dates)):
            bridge = gammabridge(m, var, 66)
            plt.plot(bridge)
        plt.xlabel('time in a trading day')
        plt.ylabel('relative volume')
        plt.title(stock_file_lst[i][:-4])
        plt.show()
        
    # plot main result: share holding curves given different initial values
    # without loss of generality, we use the last gamma bridge simulated above as our benchmark
    adjust_factors = [2, 1/2]
    for k in range(2):
        for j in range(len(time)):
            x_ = [x(time[i], bridge, t=time[j], x0=time[j]*adjust_factors[k]) for i in range(len(time)-1)]
            plt.plot(x_)
        plt.xlabel('time')
        plt.ylabel('trading volume')
        plt.title('x(t)=t*'+str(adjust_factors[k]))
        plt.show()
    
    # plot correlation between relative and total trading volume
    for i in range(len(stock_file_lst)):
        plt.plot(result_dict[stock_file_lst[i][:-4]][2])
        plt.xlabel('time intervals in a trading day')
        plt.ylabel('correlation')
    plt.title('Correlation between Relative and Total Trading Volume')
    plt.show()
       
    # plot correlation between trading volume and price changes
    for i in range(len(stock_file_lst)):
        plt.plot(result_dict[stock_file_lst[i][:-4]][3])
        plt.xlabel('trading days')
        plt.ylabel('correlation')
    plt.title('Correlation between Trading Volume and Price Changes')
    plt.show()
 
    
    