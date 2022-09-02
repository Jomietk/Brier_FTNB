



import os

os.chdir('C://Users//jomie//.spyder-py3//Fine_tuning//Fine_tuning_NB_NA_conflict')



import numpy as np


import sys
from pyhugin91 import *
from sklearn.metrics import roc_auc_score




    #%%

name_class="class"

def score_domain(name_model,name_test):
    """Build a Bayesian network and propagate evidence."""
    #L_hiden_nodes = ["C0", "C1", "C2", "C3", "CS", "CPC", "CD"]
    L_max = []
  
    try:
        
        domain = Domain.parse_domain(name_model)
        
      
       
        data_test = DataSet.parse_data_set(name_test, separator=',', error_handler=None)
        
        node_sat = domain.get_node_by_name(name_class)
       
        L_proba_1 = []

        domain.compile()




        domain.set_number_of_cases(0)
        domain.add_cases(data_test, 0, data_test.get_number_of_rows())
 
        
        L_true=[]
        L_predict=[]
        if node_sat.get_number_of_states()>2:
            for i in range(0,domain.get_number_of_cases()):
                   
                   domain.enter_case(i)
           
                   L_true.append(node_sat.get_case_state(i))       
                   
                   node_sat.retract_findings()
                   domain.propagate()
                   L_predict.append(([node_sat.get_belief(ip) for ip in range(node_sat.get_number_of_states())]))
           
                   domain.reset_inference_engine()
                   
             
            ER=roc_auc_score(L_true,L_predict,multi_class="ovo",average="weighted")     
        else:
            for i in range(0,domain.get_number_of_cases()):
                 
                 domain.enter_case(i)
         
                 L_true.append(node_sat.get_case_state(i))       
                 
                 node_sat.retract_findings()
                 domain.propagate()
                 L_predict.append(node_sat.get_belief(1))
         
                 domain.initialize()      
         
           
   
            ER=roc_auc_score(L_true,L_predict)

        domain.delete()
        data_test.delete()
    except HuginException:
        print("A Hugin Exception was raised!")
        raise
    return(ER)



def score_domain_ER(name_model,name_test):
    """Build a Bayesian network and propagate evidence."""
    #L_hiden_nodes = ["C0", "C1", "C2", "C3", "CS", "CPC", "CD"]
    L_max = []
  
    try:
        
        domain = Domain.parse_domain(name_model)
        
      
       
        data_test = DataSet.parse_data_set(name_test, separator=',', error_handler=None)
        
        node_sat = domain.get_node_by_name(name_class)
       
        L_proba_1 = []

        domain.compile()




        domain.set_number_of_cases(0)
        domain.add_cases(data_test, 0, data_test.get_number_of_rows())
 
        
        L_true=[]
        L_predict=[]
        for i in range(0,domain.get_number_of_cases()):
               
               domain.enter_case(i)
       
               L_true.append(node_sat.get_case_state(i))       
               
               node_sat.retract_findings()
               domain.propagate()
               L_predict.append(np.argmax([node_sat.get_belief(i) for i in range(node_sat.get_number_of_states())]))
       
               domain.initialize()
       
            
        #AUC = roc_auc_score(L_true, L_proba_1)
      
       
        ER=sum(np.array(L_predict)==np.array(L_true))/len(L_predict)

        domain.delete()
        data_test.delete()
    except HuginException:
        print("A Hugin Exception was raised!")
        raise
    return(ER)