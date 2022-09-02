

from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from score_domain import score_domain, score_domain_ER
from fine_tuning_missing import BFTNB
from fine_tuning_function_only import FTNB
from bcap_creation_model import model_creation
import re


from disc_entropy import MDLP_Discretizer
from sklearn.model_selection import KFold
from pyhugin91 import *
import sys
import pandas as pd
import numpy as np
import os

os.chdir('C://Users//jomie//.spyder-py3//Fine_tuning//Fine_tuning_NB_NA_conflict')

# %%

#Initialization
significant_betterBFTNB_VS_FTNB = 0
significant_worseBFTNB_VS_FTNB = 0
significant_betterFTNB_VS_NB = 0
significant_worseFTNB_VS_NB = 0
significant_betterBFTNB_VS_NB = 0
significant_worseBFTNB_VS_NB = 0


betterBFTNB_VS_FTNB = 0
worseBFTNB_VS_FTNB = 0
betterFTNB_VS_NB = 0
worseFTNB_VS_NB = 0
betterBFTNB_VS_NB = 0
worseBFTNB_VS_NB = 0
total_mean_score_FTNB= 0
total_mean_score_BFTNB= 0
total_mean_score_NB= 0


nb_epoch_total_FTNB=0
nb_epoch_total_BFTNB=0

files = os.listdir("data2//.")


L_name_data = [("data2//"+files[i]) for i in range(len(files)) if not files[i].startswith('.')]


nb_it = 100


FTNB_model = "Working_dir//FTNB_model.net"
BFTNB_model = "Working_dir//BFTNB_model.net"
NB_model = "Working_dir//NB_model.net"

name_class = "class"

L_data = []
count_data=0
#for i in range(len(L_name_data)):
    #31
#for i in range(31,32):
for i in range(10,11):
# =============================================================================
# 
#   try:
# 
# =============================================================================
    #Preprosessing and initialization
    
    name_data = L_name_data[i]
    discretization = False
    if "disc" in name_data:
        discretization = True

    L_name = list(pd.read_csv(name_data).columns)
    L_name.remove(name_class)



    name_model = f"model//{name_data[7:-4]}.net"
 
    X2 = pd.read_csv(name_data)
    X2 = X2.replace("?", np.nan)
    L = list(X2.columns)
    for i in range(len(L)):
        L[i] = re.sub('[^0-9a-zA-Z_]', '', L[i])
    X2.columns = L
    X2.to_csv(name_data, index=False)



    L_ER_fine_mean = []
    L_ER_fine_missing_mean = []
    L_ER_naive_replace_mean = []
    L_ER_naive_EM_mean = []

    L_missing2 = [0]
    L_stat = []



    L_score_FTNB = []
    L_score_BFTNB = []
    L_score_NB = []

    nb_epoch_FTNB=0
    nb_epoch_BFTNB=0

    epochaa=50
    epochab=50
    L_FA=np.zeros(epochaa)
    L_FB=np.zeros(epochab)
    L_FA_score=np.zeros(epochaa)
    L_FB_score=np.zeros(epochab)       
        











    X2 = pd.read_csv(name_data)

    
    y2 = np.array(X2[name_class])

    #The score is calculated using strtifiedShufflespli cross validation.
    sss = StratifiedShuffleSplit(n_splits=nb_it, test_size=0.2)
    #kf = KFold(n_splits=10,shuffle=False)
    #rskf = RepeatedStratifiedKFold(n_splits=10,n_repeats=10)
    for train_index, test_index in sss.split(X2, y2):


        X_train, X_test = X2.iloc[train_index, :], X2.iloc[test_index, :]

        name_train_meta = "Working_dir//train_fold_22_meta.csv"

        name_train_meta_missing = "Working_dir//train_fold_22_meta_missing.csv"

        name_test_meta = "Working_dir//test_fold_22_meta.csv"
        bins = "Working_dir//bin_entropy.csv"
        concatenation = "Working_dir//full_disc.csv"

        #Save the training and testing set of the current fold
        X_train.to_csv(name_train_meta_missing, index=False)

        X_test.to_csv(name_test_meta, index=False)



        # If needed the entropy minimization discretization algorithm is used
        if discretization:
            MDLP_Discretizer(dataset=pd.read_csv(name_train_meta_missing), testset=pd.read_csv(name_test_meta),class_label=name_class, out_path_data=name_train_meta_missing, out_test_path_data=name_test_meta,
                              out_path_bins=bins, min_bins=1, min_freq=0.01)


        #Creation of the NB model. Some of the state of the node may be rare and only be on the test set.
        #We build the model with entire dataset to have the right number of state for each node.  
        df1 = pd.read_csv(name_train_meta_missing)
        df2 = pd.read_csv(name_test_meta)
        df3 = pd.concat([df1, df2], ignore_index=True)
        df3.to_csv(concatenation, index=False)
        model_creation(concatenation, name_model)

        #Initialisation of the table with random prior. The prior table are the same of the different algorithm and change with each fold.
        domain = Domain.parse_domain(name_model)
        L_ini = []
        for i in domain.get_nodes():
            L_ini.append(np.random.randint(1,100, i.get_table().get_size()))
        domain.delete()

        #Initialisation of the parameters.
        eta = 0.01
        beta = 2
        alpha = 2
      
        #for epoch in range(1,500):
            #Fine tuning algorithm. Train the model and fine tune the CPT.
        FA=FTNB(name_model, name_train_meta_missing, L_ini, eta, alpha, beta,epochaa)
        FB=BFTNB(name_model, name_train_meta_missing, L_ini, 0.001,epochab)
        


        #Score of the different model build.
        L_score_FTNB.append(score_domain_ER(FTNB_model, name_test_meta))        
        L_score_BFTNB.append(score_domain_ER(BFTNB_model, name_test_meta))
        L_score_NB.append(score_domain_ER(NB_model, name_test_meta))
        
    
    
    
        
                 
        nb_epoch_FTNB+=FA
        nb_epoch_BFTNB+=FB
     
        
    fig = plt.figure()
    plt.plot(L_score_FTNB, label="FTBN", color="c", linewidth=5, alpha=0.7)
    plt.plot(L_score_BFTNB, label="BFTBN", color="r")
    plt.plot(L_score_NB, label="NB",
             color="g", linewidth=7, alpha=0.5)
    plt.title(name_data[7:-4])
    plt.legend()
    plt.show()
    
    count_data+=1
    
    mean_score_FTNB=np.mean(L_score_FTNB)
    mean_score_BFTNB=np.mean(L_score_BFTNB)
    mean_score_NB=np.mean(L_score_NB)
    
    
    
    total_mean_score_FTNB+=mean_score_FTNB
    total_mean_score_BFTNB+=mean_score_BFTNB
    total_mean_score_NB+=mean_score_NB
    
    L_ER_fine_mean.append(mean_score_FTNB)
    L_ER_fine_missing_mean.append(mean_score_BFTNB)
    L_ER_naive_EM_mean.append(mean_score_NB)






    # BFTNB VS FTNB

    if mean_score_BFTNB > mean_score_FTNB:
        betterBFTNB_VS_FTNB += 1
    else:
        worseBFTNB_VS_FTNB += 1

    # FTNB VS NB

    if mean_score_FTNB >mean_score_NB:
        betterFTNB_VS_NB += 1
    else:
        worseFTNB_VS_NB += 1

    # BFTNB VS NB

    if mean_score_BFTNB > mean_score_NB:
       betterBFTNB_VS_NB += 1
    else:
        worseBFTNB_VS_NB += 1






    #Calculate the t-test on two related samples of scores, a and b

    print(name_data[7:-4])
    
    
    # BFTNB VS FTNB
    r  = stats.ttest_rel(L_score_FTNB, L_score_BFTNB)
    print(r.statistic, r.pvalue)
    if r.pvalue < 0.05:
        print("FTNB VS BFTNB: significant")
        L_stat.append("significant")
    else:
        print("FTNB VS BFTNB: not significant")
        L_stat.append("not significant")

    # FTNB VS NB
    r = stats.ttest_rel(L_score_FTNB, L_score_NB)
    print(r.statistic, r.pvalue)
    if r.pvalue < 0.05:
        print("FTNB VS NB: significant")
        L_stat.append("significant")
    else:
        print("FTNB VS NB: not significant")
        L_stat.append("not significant")

    # BFTNB VS NB
    r = stats.ttest_rel(L_score_BFTNB, L_score_NB)
    print(r.statistic, r.pvalue)
    if r.pvalue < 0.05:
        print("BFTNB VS NB: significant")
        L_stat.append("significant")
    else:
        print("BFTNB VS NB: not significant")
        L_stat.append("not significant")

    print("FTNB", np.round(mean_score_FTNB, 3), "BFTNB", np.round(mean_score_BFTNB, 3), "NB", np.round(mean_score_NB, 3))

    # BFTNB VS FTNB
    if L_stat[0] == "significant":
        if mean_score_BFTNB > mean_score_FTNB:
            significant_betterBFTNB_VS_FTNB += 1
        else:
            significant_worseBFTNB_VS_FTNB += 1

    # FTNB VS NB
    if L_stat[1] == "significant":
        if mean_score_FTNB > mean_score_NB:
            significant_betterFTNB_VS_NB += 1
        else:
            significant_worseFTNB_VS_NB += 1

    # BFTNB VS NB
    if L_stat[2] == "significant":
        if mean_score_BFTNB > mean_score_NB:
            significant_betterBFTNB_VS_NB += 1
        else:
            significant_worseBFTNB_VS_NB += 1





    #Creation of the dataset with score and significancy
    L_perc = np.array([ np.round(mean_score_BFTNB, 3)*100,  np.round(mean_score_FTNB, 3)* 100, np.round(mean_score_NB, 3)*100, L_stat[0], L_stat[1], L_stat[2]])
    f = np.transpose(pd.DataFrame(L_perc))
    f.columns = ["BFTNB", "FTNB", "NB","FTNB VS BFTNB", "FTNB VS NB", "BFTNB VS NB"]
    f = f.T
    f["Step"]=[nb_epoch_BFTNB/nb_it,nb_epoch_FTNB/nb_it,"","","",""]
    
    nb_epoch_total_FTNB+=nb_epoch_FTNB/nb_it
    nb_epoch_total_BFTNB+=nb_epoch_BFTNB/nb_it
    
    g = pd.DataFrame(data=np.matrix([name_data[7:-4]]), index=["data"])
    L_data.append(g)
    L_data.append(f)
# =============================================================================
#   except:
#      pass
# 
# =============================================================================

result = pd.concat(L_data)
result.rename(columns = {0:eta}, inplace = True)
result.to_csv("Result.csv")



k=pd.DataFrame({"Step":[betterBFTNB_VS_FTNB,betterFTNB_VS_NB,betterBFTNB_VS_NB]}, index=["betterBFTNB_VS_FTNB","betterFTNB_VS_NB","betterBFTNB_VS_NB"])
p = pd.DataFrame({"Step":[significant_betterBFTNB_VS_FTNB,significant_betterFTNB_VS_NB,significant_betterBFTNB_VS_NB]}, index=["significant_betterBFTNB_VS_FTNB","significant_betterFTNB_VS_NB","significant_betterBFTNB_VS_NB"])

v=pd.DataFrame({"Step":[worseBFTNB_VS_FTNB,worseFTNB_VS_NB,worseBFTNB_VS_NB]}, index=["worseBFTNB_VS_FTNB","worseFTNB_VS_NB","worseBFTNB_VS_NB"])
b = pd.DataFrame({"Step":[significant_worseBFTNB_VS_FTNB,significant_worseFTNB_VS_NB,significant_worseBFTNB_VS_NB]}, index=["significant_worseBFTNB_VS_FTNB","significant_worseFTNB_VS_NB","significant_worseBFTNB_VS_NB"])


    
d = pd.DataFrame({"Step":[total_mean_score_FTNB/count_data,total_mean_score_BFTNB/count_data,total_mean_score_NB/count_data,np.round(nb_epoch_total_BFTNB/count_data,1),np.round(nb_epoch_total_FTNB/count_data,1),count_data]}, index=["total_mean_score_FTNB","total_mean_score_BFTNB","total_mean_score_NB","mean_nb_epoch_BFTNB","mean_nb_epoch_FTNB","nb_data"])



KP=pd.concat([k,p,v,b,d])
KP_result=pd.concat([result,KP])
KP_result.to_csv("Result_full_test.csv")




print(significant_betterBFTNB_VS_FTNB,significant_betterFTNB_VS_NB,significant_betterBFTNB_VS_NB)
print(significant_worseBFTNB_VS_FTNB,significant_worseFTNB_VS_NB,significant_worseBFTNB_VS_NB)




