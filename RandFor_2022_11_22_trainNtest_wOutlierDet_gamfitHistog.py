# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:35:52 2022

@author: bgoh
"""
# -*- coding: utf-8 -*-
"""
Created on Fri, 5 May 2022, 13:0 0h

Feature dimensional reduction:
   1) Bifurcation: train set / test set
   2 )Encapsulation of MLtrain & MLtest routines
   3) Fit unpruned feature set to test set --> report MAPE
   4) Spearman prune, then fit to reduced feature set --> report MAPE
   5) Randseed perturubation prune, then fit to 2' reduced feature set --> report MAPE
   6) Test on 2' pruned feature set. 
   7) Plots: 
        Scatterplot of train (DOES NOT TEST)
        Histogram of errors
        Barplot of feature ordinal rankings < Randomness Feature 
        Barplot comparison of MAPEs between: unpruned feature set, Spearman-pruned only, Randseed perturbation pruned 

@author: bgoh, JHS
"""
#%% (1) Import all necessary Packages
# import sys
#sys.modules[__name__].__dict__.clear() #start with a clean workspace
import time
startTime = time.time()

# import re
import numpy as np
import pandas as pd
# import pymatgen as mg
from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from sklearn.linear_model import LinearRegression
# from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import sklearn.model_selection as xval
# from sklearn import datasets
# from sklearn import decomposition
# from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import mean_squared_error
from math import sqrt

import pathlib


from scipy import stats
from openpyxl import Workbook
from random import sample
# from itertools import combinations

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#%% (2) Input files & specification params
normScale = StandardScaler()
dataFile = 'ML_InpAsOf_2022_12_20_ICPMS_all.xlsx'
allData_whichSheet = 'Sheet1'
#testData_whichSheet = 'Pl23'

randSeedVal = 1
modelType = "randFor Spearman-randSeedPerturb 2prune train xSc yUnsc"
targetVariableType = 'saltConc_ppm_ICPMS_'
elem = 'total'
target_var = 'saltConc_ppm_ICPMS_'+elem

unit = "[ppm]"
normScale = StandardScaler()#it's a normscaler

SS316CorrCriteria = 1194 #[ppm]
SS316CorrCriteria_uncert = 163
resultsExportDir = 'LatestOutput/Manuscript_graphs/TrainTest/'


#%% (3) Homebrew functions
"""
Define the function that will interrogate the individual trees in the RF
jrhs wrote this function
"""


def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X)[x])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up
    
def MLtrainsetMetrics(model,X_inps,y_inps):
    X_inps_np = X_inps.to_numpy()
    y_inps_np = y_inps.to_numpy()
    model.fit(X_inps_np,y_inps_np)
    Rsq = model.score(X_inps_np,y_inps_np,sample_weight=None)
    y_fit = model.predict(X_inps_np)
    RMSE_fit = sqrt(mean_squared_error(y_inps_np, y_fit))
    
    uncert_down_y_predict, uncert_up_y_predict = pred_ints(model,X_inps_np, percentile=90)
    uncert_down_y_predict_np = np.array(uncert_down_y_predict)
    uncert_up_y_predict_np = np.array(uncert_up_y_predict)
    uncertMargin_y_pred = (uncert_up_y_predict_np - uncert_down_y_predict_np)/2
	
    yfit_xvalPredict = model_selection.cross_val_predict(model,X_inps_np,y_inps_np,cv=5)
    Rsq_xval = model_selection.cross_val_score(model,X_inps_np,y_inps_np,cv=5)
    RMSE_xval = sqrt(mean_squared_error(y_inps_np, yfit_xvalPredict))
    y_inps_arr = np.array(y_inps)
    yfit_xvalPredict_arr = np.array(yfit_xvalPredict)
    yfit_absPredErrs = abs(yfit_xvalPredict_arr - y_inps_arr)
    yfit_percPredErrs = np.divide(yfit_absPredErrs,abs(y_inps_arr))*100 #y prediction errors, units [%]
    yfit_xvalMAPE = yfit_percPredErrs.mean()
    yfit_xvalMAPE_stdev = yfit_percPredErrs.std()
    featImpts = model.feature_importances_
    return Rsq,Rsq_xval, y_fit, RMSE_fit, uncert_down_y_predict_np, uncert_up_y_predict_np, uncertMargin_y_pred, yfit_xvalPredict, RMSE_xval, yfit_percPredErrs, yfit_xvalMAPE, yfit_xvalMAPE_stdev, featImpts

def MLtestsetMetrics(model,X_inps_train,y_inps_train,X_inps_test,y_inps_test):
    X_inps_train_np = X_inps_train.to_numpy()
    y_inps_train_np = y_inps_train.to_numpy()
    model.fit(X_inps_train_np,y_inps_train_np)
    X_inps_test_np = X_inps_test.to_numpy()
    y_test_pred = model.predict(X_inps_test_np)
    Rsq_test = model.score(X_inps_test,y_inps_test,sample_weight=None)
    #y_fit = model.predict(X_inps)
    #RMSE_fit = sqrt(mean_squared_error(y_inps, y_fit))
    
    #uncert_down_y_predict, uncert_up_y_predict = pred_ints(model,X_inps, percentile=90)
    #uncert_down_y_predict_np = np.array(uncert_down_y_predict)
    #uncert_up_y_predict_np = np.array(uncert_up_y_predict)
    #uncertMargin_y_pred = (uncert_up_y_predict_np - uncert_down_y_predict_np)/2
	
    #yfit_xvalPredict = model_selection.cross_val_predict(model,X_inps,y_inps,cv=5)
    #RMSE_xval = sqrt(mean_squared_error(y_inps, yfit_xvalPredict))
    #y_inps_arr = np.array(y_inps)
    y_fit_arr = np.array(y_test_pred)
    yfit_absPredErrs = abs(y_fit_arr - y_inps_test)
    yfit_percPredErrs = np.divide(yfit_absPredErrs,abs(y_inps_test))*100 #y prediction errors, units [%]
    yfit_MAPE = yfit_percPredErrs.mean()
    yfit_MAPE_stdev = yfit_percPredErrs.std()
    return  Rsq_test,y_test_pred, yfit_percPredErrs, yfit_MAPE, yfit_MAPE_stdev


df_total = pd.read_excel(dataFile, sheet_name = allData_whichSheet)#),sheet_name="Sheet3")
df_total = df_total.dropna(axis=0)

#the scaled set
df_colnames = df_total.columns
df_total_scaled = normScale.fit_transform(df_total)
df_total_scaled = pd.DataFrame(df_total_scaled,columns = df_colnames)

numSamples_total = df_total.shape[0]
numSamples_test = int(round(0.2*numSamples_total,0))
samplesIndexList = list(range(numSamples_total))
ytest_pred_MAPE= 100 #starting value of MAPE is arbitrarily large
iterCount = 0

#%% (4) start sequence

while ytest_pred_MAPE > 60: #truePositives <0.8: #
    
    
    testSampleIDs = [103,46,34,54,3,106,67,61,43,56,27,72,53,51,40,26,6,38,13,21,57,45]#sample(list(range(numSamples_total)),numSamples_test)#[103,32,65,104,19,69,50,42,51,64,96,70,55,75,78,16,52,15,100,38,28,68] #
    

    
    
    df_train = df_total.drop(df_total.index[testSampleIDs])
    df_test = df_total.loc[testSampleIDs]
    df_train_scaled = df_total_scaled.drop(df_total.index[testSampleIDs])
    df_test_scaled = df_total_scaled.loc[testSampleIDs]

    ytrain = df_train[target_var] #target col name GDOES_saltConc_Fe_mols
    ytest = df_test[target_var]
    
    Xtrain_scaled = df_train_scaled.drop(['PlateID', 
                                     'Sample#',
                                     targetVariableType+'Cr',
                                     targetVariableType+'Fe',
                                     targetVariableType+'Mn',
                                     targetVariableType+'Ni',
                                     targetVariableType+'total'],axis='columns')#
    
    Xtest_scaled = df_test_scaled.drop(['PlateID', 
                                     'Sample#',
                                     targetVariableType+'Cr',
                                     targetVariableType+'Fe',
                                     targetVariableType+'Mn',
                                     targetVariableType+'Ni',
                                     targetVariableType+'total'],axis='columns')#
    
    numSamples_train = df_train.shape[0]
    
    ytrain_exp = ytrain
    Xtrain_unpruned_exp = Xtrain_scaled#randomness feature is always the 1st feature in the X-matrix

    """
    (4b) Get List of Features for everyone. We assume the cols are exactly the same for X_train & X_test. 
    that's why you format the cols. first using the mastercopy then split only on rows. 
    """
    X_len = len(Xtrain_unpruned_exp)#number of samples
    X_names_unpruned = Xtrain_unpruned_exp.columns #Xnames
    X_names_unpruned_arr = np.transpose(np.array([X_names_unpruned]))
    numFeats_unpruned = np.size(X_names_unpruned_arr)
    
    model_randFor = RandomForestRegressor(n_estimators=1000, random_state = 1) #initialize hyperparams for model_randfor
    unpruned_train_Rsq, unpruned_train_Rsq_xval, unpruned_train_y_fit,unpruned_train_RMSE_fit,unpruned_train_uncert_down_y_predict_np, unpruned_train_uncert_up_y_predict_np, unpruned_train_uncertMargin_y_pred, unpruned_train_yfit_xvalPredict,unpruned_train_RMSE_xval,unpruned_train_yfit_percPredErrs,unpruned_train_yfit_xvalMAPE,unpruned_train_yfit_xvalMAPE_stdev,unpruned_train_featImpts=MLtrainsetMetrics(model_randFor,Xtrain_unpruned_exp,ytrain_exp)
    
    numSamples_test = df_test.shape[0]
    ytest_exp = ytest
    Xtest_exp = Xtest_scaled

    Rsq_test, y_test_predict, ytest_percPredErrs, ytest_pred_MAPE, ytest_pred_MAPE_stdev =MLtestsetMetrics(model_randFor,Xtrain_unpruned_exp,ytrain_exp, Xtest_scaled,ytest_exp)
    expectedGoodCorrSampleIDs = np.where(y_test_predict<SS316CorrCriteria)
    df_expectedGoodCorrSamples = df_test.iloc[expectedGoodCorrSampleIDs]
    ytest_goodCorr_exp = df_expectedGoodCorrSamples[target_var]
    ytest_goodCorr_pred = y_test_predict[expectedGoodCorrSampleIDs]
    numSamples_test_selected = df_expectedGoodCorrSamples.shape[0]
        
    df_expectedBadCorrSamples = df_test.drop(df_test.index[expectedGoodCorrSampleIDs])
    ytest_badCorr_exp = df_expectedBadCorrSamples[target_var]
    ytest_pred_df = pd.DataFrame(y_test_predict)
    ytest_badCorr_pred = ytest_pred_df.drop(ytest_pred_df.index[expectedGoodCorrSampleIDs])
    
    ytest_goodCorr_exp_arr = np.array(ytest_goodCorr_exp)
    ytestGoodCorr_absPredErrs = abs(ytest_goodCorr_pred - ytest_goodCorr_exp_arr)
    ytest_goodCorr_percPredErrs = np.divide(ytestGoodCorr_absPredErrs,abs(ytest_goodCorr_exp_arr))*100 #y prediction errors, units [%]
    ytest_goodCorr_MEANpercErr = ytest_percPredErrs.mean()
    ytest_goodCorr_STDEVpercErr = ytest_percPredErrs.std()
    print('overall test MAPE = ' + str(round(ytest_pred_MAPE,1))+'%')
    iterCount = iterCount+1
    #print('good corr MAPE = ' + str(round(ytest_goodCorr_MEANpercErr,1))+'%' + "+/-" + str(round(ytest_goodCorr_STDEVpercErr,0)) + " %" )


print('iteration ended @: #' + str(iterCount))
#%%  (5) sorting sampels into confusion matrix
"""
(5) Confusion Matrix
"""
#percSelected = numSamples_test_selected/numSamples_test
test_reject = np.where(y_test_predict>SS316CorrCriteria)
test_shouldBselect = np.where(ytest_exp<SS316CorrCriteria)
test_shouldBreject = np.where(ytest_exp>SS316CorrCriteria)
confus_TP = np.intersect1d(expectedGoodCorrSampleIDs,test_shouldBselect)
sampleID_TP = [testSampleIDs[i] for i in confus_TP]
confus_FN = np.intersect1d(test_shouldBselect,test_reject)
sampleID_FN = [testSampleIDs[i] for i in confus_FN]
confus_FP = np.intersect1d(expectedGoodCorrSampleIDs,test_shouldBreject)
sampleID_FP = [testSampleIDs[i] for i in confus_FP]
confus_TN = np.intersect1d(test_reject,test_shouldBreject)
sampleID_TN = [testSampleIDs[i] for i in confus_TN]
numTP = len(confus_TP)
numFN = len(confus_FN)
numFP = len(confus_FP)
numTN = len(confus_TN)
confusionVector = np.zeros(numSamples_total)
for j in sampleID_TP:
    confusionVector[j] =1
for j in sampleID_FN:
    confusionVector[j] =2
for j in sampleID_FP:
    confusionVector[j] =3
for j in sampleID_TN:
    confusionVector[j] =4

#%%  
"""
(5b) Outlier Detection
"""
from sklearn.ensemble import IsolationForest

clf = IsolationForest(random_state=1).fit(Xtrain_unpruned_exp)
testSetOutlierVect = clf.predict(Xtest_exp)

#testSet_TrainDisb_in = np.where(testSetOutlierVect==1)
testSet_TrainDisb_out = np.where(testSetOutlierVect==-1)
testSet_TrainDisb_out_lst = list(testSet_TrainDisb_out)
testSet_TrainDisb_out_yexp = [ytest_exp.iloc[i] for i in testSet_TrainDisb_out_lst]
testSet_TrainDisb_out_ypred = [y_test_predict[i] for i in testSet_TrainDisb_out]

#%% (6) Plotting
"""
(6) Plotting module
"""
plt.rcParams['figure.figsize'] = [6, 6]
fig1 = plt.figure()
plt.plot(ytrain_exp,ytrain_exp,color='blue',label='parity plot')#doing plt.plot or plt.scatter in succession overlays the plot by default.
#feature importances for the model of each elem
randForUncertBand_df = pd.DataFrame(np.column_stack((ytrain_exp,unpruned_train_uncert_up_y_predict_np,unpruned_train_uncert_down_y_predict_np)))
randForUncertBand_df_sorted = randForUncertBand_df.sort_values(0,ascending=True)#1st arg is the col which you're sorting by
randForUncertBand_arr_sorted = np.array(randForUncertBand_df_sorted)
plt.fill_between(randForUncertBand_arr_sorted[:,0], randForUncertBand_arr_sorted[:,1],randForUncertBand_arr_sorted[:,2], alpha=.5, linewidth=0)
plt.scatter(ytrain_exp,unpruned_train_y_fit,label='training set ('+str(numSamples_train)+'), xval MAPE='+str(round(unpruned_train_yfit_xvalMAPE,1))+'%',color='green')
plt.scatter(ytest_exp,y_test_predict,label='test set ('+str(numSamples_test)+'), MAPE ='+ str(round(ytest_pred_MAPE,1))+ " %", color='orange',edgecolors='black')
# plt.scatter(ytest_goodCorr_exp,ytest_goodCorr_pred, label='test set selected ('+str(numSamples_test_selected)+')' , color = 'red')
# plt.scatter(ytest_badCorr_exp,ytest_badCorr_pred, label='test set rejected', color = 'black')
# plt.scatter(testSet_TrainDisb_out_yexp, testSet_TrainDisb_out_ypred, label='model outliers', s=200, edgecolors='purple',c = 'none')
# plt.title("test samples: " + str(testSampleIDs))
plt.xlabel("Experimental value" + unit)
plt.ylabel("Random Forest predicted value" + unit)

# # this segment is for the confusion matrix
# plt.hlines(SS316CorrCriteria,0,3500,linestyles='dashed')
# plt.vlines(SS316CorrCriteria,0,3500,linestyles='dashed')
# plt.text(0,0,"true positives")
# plt.text(SS316CorrCriteria,0,"false positives")
# plt.text(SS316CorrCriteria,2500,"true negatives")
# plt.text(0,2500,"false negatives")

# plt.fill_between(np.linspace(0,3500,10) , np.ones(10)*SS316CorrCriteria - SS316CorrCriteria_uncert, np.ones(10)*SS316CorrCriteria + SS316CorrCriteria_uncert, linewidth=0,color=colors[1], alpha=0.2)
# plt.fill_between(np.linspace(SS316CorrCriteria-SS316CorrCriteria_uncert,SS316CorrCriteria+SS316CorrCriteria_uncert,10) ,np.zeros(10),np.ones(10)*3500, linewidth=0,color=colors[1], alpha=0.2)


plt.legend()
plt.show()#displays the plot in the console
parityPlotName = "rfr_prediction_e.png"
fig1.savefig(resultsExportDir +  parityPlotName)#save plot

#%% histogram with gamma pdf
plt.rcParams['figure.figsize'] = [6.4, 4]
fig2, ax1 = plt.subplots()

# ax1.hist(unpruned_train_yfit_percPredErrs,bins=15, density=True, color = colors[0], edgecolor='black', alpha = 1, linewidth=1.2, label='train set fitting %errs, mean='+str(round(unpruned_train_yfit_xvalMAPE,1))+'%')
# ax1.hist(ytest_percPredErrs,bins=15, density=True, color = colors[1], edgecolor='black', alpha = 0.7, linewidth=1.2, label = 'test set pred %errs, mean='+str(round(ytest_pred_MAPE,1))+'%')#%errs
UTY_ppe = unpruned_train_yfit_percPredErrs/100
YT_ppe = ytest_percPredErrs/100
ax1.hist(UTY_ppe,bins=25, density=True, color = colors[0], edgecolor='black', alpha = 1, linewidth=1.2, label='train set fitting %errs, mean='+str(round(unpruned_train_yfit_xvalMAPE,1))+'%')
ax1.hist(YT_ppe,bins=25, density=True, color = colors[1], edgecolor='black', alpha = 0.7, linewidth=1.2, label = 'test set pred %errs, mean='+str(round(ytest_pred_MAPE,1))+'%')#%errs

# import matplotlib.ticker as mtick
# ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.minorticks_on()
ax1.tick_params(axis='y', which='minor', left=False)

ax1.set_ylabel('probability density')
plt.xlabel("abs % err between Model Prediction & Experimental value [x100 %]")

from scipy import stats
from scipy.special import gamma
from scipy.stats import gamma
from scipy.optimize import curve_fit

# ax2=ax1.twinx()
# ax2.set_ylabel("Probability")

fit_alpha, fit_loc, fit_beta = stats.gamma.fit(unpruned_train_yfit_percPredErrs)
test_alpha, test_loc, test_beta = stats.gamma.fit(ytest_percPredErrs)

histoLinspace_size = 100#int(max(unpruned_train_yfit_percPredErrs))
# histoLinspace = np.linspace(gamma.pdf(0.01, fit_alpha),gamma.pdf(0.99, fit_alpha), histoLinspace_size)
histoLinspace = np.linspace(max(unpruned_train_yfit_percPredErrs),gamma.pdf(0.99, fit_alpha), histoLinspace_size)
# unpruned_train_yfit_percPredErrs_gamfit = stats.gamma.pdf(histoLinspace,fit_alpha,scale=fit_beta)
# bin_heights, bin_borders, _ = plt.hist(unpruned_train_yfit_percPredErrs_gamfit,bins=15,label='bin heights n borders')#int(max(unpruned_train_yfit_percPredErrs)+1))
# bin_centers = bin_borders[:-1]+ np.diff(bin_borders)/2
histoGam_fit = stats.gamma.pdf(histoLinspace,a=fit_alpha,scale=fit_beta)
histoGam_test = stats.gamma.pdf(histoLinspace,a=test_alpha,scale=test_beta)

# ax2.plot(histoLinspace,histoGam_fit,color='black',label='training set gamma fit pdf')
# ax2.plot(histoLinspace,histoGam_test,color='black',label='test set gamma fit pdf')
# plt.title("Model type: " + modelType + " Target: "+ elem)
# plt.figtext(0.6,0.7,"train MAPE= ") #the x,y coords represent "relative proportion of xy axes".
# plt.figtext(0.6,0.65,"test MAPE= ") 


plt.legend()
plt.show()
hisPlotName = modelType+"_"+elem+"_%errHistoGam.png"
fig2.savefig(resultsExportDir + hisPlotName)

#%% only gamma pdf
plt.rcParams['figure.figsize'] = [6.4, 4]
fig3 = plt.figure()

from scipy import stats
from scipy.special import gamma
from scipy.stats import gamma
from scipy.optimize import curve_fit

fit_alpha, fit_loc, fit_beta = stats.gamma.fit(unpruned_train_yfit_percPredErrs)
test_alpha, test_loc, test_beta = stats.gamma.fit(ytest_percPredErrs)

histoLinspace_size = 100#int(max(unpruned_train_yfit_percPredErrs))
histoLinspace = np.linspace(gamma.pdf(0.01, fit_alpha),gamma.pdf(0.99, fit_alpha), histoLinspace_size)
# unpruned_train_yfit_percPredErrs_gamfit = stats.gamma.pdf(histoLinspace,fit_alpha,scale=fit_beta)
# bin_heights, bin_borders, _ = plt.hist(unpruned_train_yfit_percPredErrs_gamfit,bins=15,label='bin heights n borders')#int(max(unpruned_train_yfit_percPredErrs)+1))
# bin_centers = bin_borders[:-1]+ np.diff(bin_borders)/2
histoGam_fit = stats.gamma.pdf(histoLinspace,a=fit_alpha,scale=fit_beta)
histoGam_test = stats.gamma.pdf(histoLinspace,a=test_alpha,scale=test_beta)

plt.plot(histoLinspace,histoGam_fit,color='blue',label='training set pdf')
plt.plot(histoLinspace,histoGam_test,color='orange',label='test set pdf')
# plt.title("Model type: " + modelType + " Target: "+ elem)
# plt.figtext(0.6,0.7,"train MAPE= ") #the x,y coords represent "relative proportion of xy axes".
# plt.figtext(0.6,0.65,"test MAPE= ") 

plt.xlabel("abs % err between Model Prediction & Experimental value [%]")
plt.ylabel("Probability")
plt.legend()
plt.show()
hisPlotName = modelType+"_"+elem+"_%errGamOnly.png"
fig2.savefig(resultsExportDir + hisPlotName)
#%% (6a) Scatterplot: xval
plt.rcParams['figure.figsize'] = [6, 6]
fig6 = plt.figure()
plt.plot(ytrain_exp,ytrain_exp,color='blue',label='parity plot')#doing plt.plot or plt.scatter in succession overlays the plot by default.
#feature importances for the model of each elem
randForUncertBand_df = pd.DataFrame(np.column_stack((ytrain_exp,unpruned_train_uncert_up_y_predict_np,unpruned_train_uncert_down_y_predict_np)))
randForUncertBand_df_sorted = randForUncertBand_df.sort_values(0,ascending=True)#1st arg is the col which you're sorting by
randForUncertBand_arr_sorted = np.array(randForUncertBand_df_sorted)
plt.fill_between(randForUncertBand_arr_sorted[:,0], randForUncertBand_arr_sorted[:,1],randForUncertBand_arr_sorted[:,2], alpha=0.2, linewidth=0, label = 'random forest uncertainty band twice-feature pruned')
plt.scatter(ytrain_exp,unpruned_train_yfit_xvalPredict,label='unpruned training set ('+str(numSamples_train)+')',edgecolors='green',c = 'none')
plt.scatter(ytest_goodCorr_exp,ytest_goodCorr_pred, label='test set selected ('+str(numSamples_test_selected)+')' , color = 'red')
plt.scatter(ytest_badCorr_exp,ytest_badCorr_pred, label='test set rejected', color = 'black')
# plt.title("test samples: " + str(testSampleIDs))
plt.xlabel("Experimental value" + unit)
plt.ylabel("Random Forest XVAL predicted value" + unit)

# this segment is for the confusion matrix
plt.hlines(SS316CorrCriteria,0,3500,linestyles='dashed')
plt.vlines(SS316CorrCriteria,0,3500,linestyles='dashed')
plt.text(0,0,"true positives")
plt.text(SS316CorrCriteria,0,"false positives")
plt.text(SS316CorrCriteria,2400,"true negatives")
plt.text(0,2400,"false negatives")
plt.text(SS316CorrCriteria+1000,SS316CorrCriteria,"y test MAPE = " + str(round(ytest_pred_MAPE,0)) + "+/-" + str(round(ytest_pred_MAPE_stdev,0)) + " %")



# plot n save the scatter plot
plt.legend()
plt.show()#displays the plot in the console
parityPlotName = modelType+"_"+elem+"_RFparityPlot.png"
fig6.savefig(resultsExportDir +" XVAL "+  parityPlotName)#save plot

   

executionTimeInMins = (time.time() - startTime)/60
print('Execution time in mins = ' + "{:.2f}".format(executionTimeInMins))
