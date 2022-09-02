







import numpy as np
import sys
from pyhugin91 import *
from create_table import creat_table
import os
from score_domain import score_domain, score_domain_ER

os.chdir('C://Users//jomie//.spyder-py3//Fine_tuning//Fine_tuning_NB_NA_conflict')





    #%%
    
    
def brier_multi(targets, probs):
    targets = np.array(targets)
    probs = np.array(probs)
    return np.mean(np.sum((probs - targets)**2, axis=1))


FTNB_model = "Working_dir//FTNB_model.net"
BFTNB_model = "Working_dir//BFTNB_model.net"
NB_model = "Working_dir//NB_model.net"

name_class="class"







name_train_meta = "Working_dir//train_fold_22_meta.csv"
name_train_meta_missing = "Working_dir//train_fold_22_meta_missing.csv"
name_test_meta = "Working_dir//test_fold_22_meta.csv"



def FTNB(name_model,data_name,L_ini,eta,alpha,beta,epoch):

          
            try:
               
                L=[]
                L_score=[]
                data=DataSet.parse_data_set(data_name, separator=',', error_handler=None)
                
                domain = Domain.parse_domain(name_model)
                L_nodes=domain.get_nodes()

                ER=0        
                node_class=domain.get_node_by_name(name_class)      
                nb_class_node_class=node_class.get_number_of_states()
               
                #Initialization of the CPT with random prior distribution
                for i in range(len(L_nodes)):
                     L_nodes[i].get_table().set_data(L_ini[i])     
             
                domain.add_cases(data,0,data.get_number_of_rows())                               
                domain.compile()      
                
                #Creation of the experience tables
                for i in L_nodes:
                    size_table=i.get_experience_table().get_size()    
                    i.get_experience_table().set_data([1/size_table]*size_table)
                    

                    
                domain.learn_tables()
                
                #Save the trained NB model 
                domain.save_as_net(NB_model)
        
                     
                domain.set_number_of_cases(0)                 
               
                domain.add_cases(data,0, data.get_number_of_rows())   
                 
                L_nodes=node_class.get_children()       
                
                
                
                #Estimation of the score of the NB.
                L_actuall=np.zeros(domain.get_number_of_cases())
                L_predict=np.zeros(domain.get_number_of_cases())
                for i in range(0,domain.get_number_of_cases()):
                        
                    domain.enter_case(i)          
                        
                        
                    c_actuall=node_class.get_case_state(i)
                        
                    L_actuall[i]=c_actuall
                        
                    node_class.retract_findings()
                    domain.propagate()
                    c_predict=np.argmax([node_class.get_belief(i) for i in range(node_class.get_number_of_states())])
                        
                    L_predict[i]=c_predict
                        
                       
                    domain.initialize()
                
                
                score=sum(L_predict==L_actuall)/len(L_actuall)
                
  
                        
                #Fine tuning algorithm

                max_score=0
                nb_step=-1
                while(score>max_score):
                #for gh in range(epoch):
                    nb_step+=1
                    domain.save_as_net(FTNB_model) 
                    max_score=score
                    for i in range(0,domain.get_number_of_cases()):
                        
                        domain.enter_case(i)           
                        c_actuall=node_class.get_case_state(i) 
                        node_class.retract_findings()
                        domain.propagate()
                        c_predict=np.argmax([node_class.get_belief(jk) for jk in range(node_class.get_number_of_states())])
                        domain.initialize()
                        
                        if c_predict!=c_actuall:
                            error=abs(node_class.get_belief(c_actuall)-node_class.get_belief(c_predict))                            
                            for node in L_nodes:
                                if node.case_is_set(i):
                                    ai=node.get_case_state(i)
                                    #Creation of the fine tune CPT for a node
                                    table=creat_table(node,i,c_actuall,c_predict,eta,alpha,beta,error,nb_class_node_class,ai)                                    
                                    node.get_table().set_data(table,0,len(table))
                             
                                    #Normalized cpt
                                    domain.enter_case(0)
                                    domain.propagate()
                                    domain.initialize
                                    
                                    

                                
                    # Calculation of the score after a of fine tuning epoch
    
                    L_actuall=np.zeros(domain.get_number_of_cases())
                    L_predict=np.zeros(domain.get_number_of_cases())
                    s=0
                    for jk in range(0,domain.get_number_of_cases()):
                        
                        domain.enter_case(jk)          
                
                
                        c_actuall=node_class.get_case_state(jk)
                
                        L_actuall[jk]=c_actuall
                
                        node_class.retract_findings()
                        domain.propagate()
                        c_predict=np.argmax([node_class.get_belief(jkp) for jkp in range(node_class.get_number_of_states())])
                
                        L_predict[jk]=c_predict
                
                        s+=(c_predict==c_actuall)*1
                        domain.initialize()
                    

                #Delete domain and data for memory
                domain.delete()
                data.delete()
            except HuginException:
                print("A Hugin Exception was raised!")
                raise
            return nb_step
         
            
             
           
