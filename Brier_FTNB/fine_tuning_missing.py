



from sklearn.metrics import roc_auc_score


from score_domain import score_domain, score_domain_ER

from pyhugin91 import *
import sys
import numpy as np
import os

os.chdir('C://Users//jomie//.spyder-py3//Fine_tuning//Fine_tuning_NB_NA_conflict')

# %%

def sig(x):
    return 1/(1+np.exp(-x))


def brier_multi(targets, probs):
    targets = np.array(targets)
    probs = np.array(probs)
    return np.mean(np.sum((probs - targets)**2, axis=1))


name_class = "class"



FTNB_model = "Working_dir//FTNB_model.net"
BFTNB_model = "Working_dir//BFTNB_model.net"





name_train_meta = "Working_dir//train_fold_22_meta.csv"
name_train_meta_missing = "Working_dir//train_fold_22_meta_missing.csv"
name_test_meta = "Working_dir//test_fold_22_meta.csv"





def BFTNB(name_model, data_name, L_ini, eta,epoch):

    try:
        L=[]
        data = DataSet.parse_data_set(
            data_name, separator=',', error_handler=None)

        domain = Domain.parse_domain(name_model)
        L_nodes = domain.get_nodes()

        L_auc = []
        AUC = 0
        ER = 0

        node_sat = domain.get_node_by_name(name_class)
        nb_class_node_sat = node_sat.get_number_of_states()

        # Initialize the CPT randomly (same initialization for the different algorithm)
        for i in range(len(L_nodes)):
            L_nodes[i].get_table().set_data(L_ini[i])


        domain.add_cases(data, 0, data.get_number_of_rows())

        domain.compile()

        #Initialize experience table
        for i in L_nodes:
            size_table = i.get_experience_table().get_size()
            i.get_experience_table().set_data([1/size_table]*size_table)


        domain.learn_tables()


        domain.set_number_of_cases(0)

        domain.add_cases(data, 0, data.get_number_of_rows())

        L_nodes = node_sat.get_children()





        #Calculation of the original Brier socre of the NB
        L_true_total = []
        L_predict_total = []
        for i in range(0, domain.get_number_of_cases()):

            domain.enter_case(i)

            L_true = [0]*node_sat.get_number_of_states()
            L_true[node_sat.get_case_state(i)] = 1

            L_true_total.append(L_true)

            node_sat.retract_findings()
            domain.propagate()

            L_predict = [node_sat.get_belief(oi) for oi in range(node_sat.get_number_of_states())]
            L_predict_total.append(L_predict)
            domain.initialize()

        score = brier_multi(L_true_total, L_predict_total)

        #nb_nodes=len(L_nodes)

        # Finue tuning algorithm with Brier score
        min_score = 2
        nb_step=-1
        L_score=[]
        while(score < min_score): 
        #for hg in range(epoch):
            domain.save_as_net(BFTNB_model)
            min_score = score
            nb_step+=1      
            
            for i in range(0, domain.get_number_of_cases()):

                domain.enter_case(i)

                c_actuall = node_sat.get_case_state(i)
                node_sat.retract_findings()
                domain.propagate()
                c_predict = np.argmax([node_sat.get_belief(jk) for jk in range(node_sat.get_number_of_states())])
           
              
                domain.initialize()

                #Fine tune only on missclassified instance
                if c_predict != c_actuall:
  
                    for node in L_nodes:
                            # Ignore the node if the value is missing
                            if node.case_is_set(i): 



                                    ai=node.get_case_state(i)
                                    nb_states=node.get_number_of_states()
                                    table=node.get_table().get_data()
                                    table[c_actuall*nb_states+ai]=table[c_actuall*nb_states+ai]+eta#*sig(d) #*table[c_actuall*nb_states+ai]
        
                                    if table[c_predict*nb_states+ai]-eta>0:
                                        table[c_predict*nb_states+ai]=table[c_predict*nb_states+ai]-eta#*sig(d) #*table[c_predict*nb_states+ai]
                                        
                                    node.get_table().set_data(table, 0, len(table))

                                    
               
                                    #Normalized cpt
                                    domain.enter_case(0)
                                    domain.propagate()
                                    domain.initialize()
        



            # Estimate the Brier score after one epoch of fine tunning
            L_true_total = []
            L_predict_total = []
            for i in range(0, domain.get_number_of_cases()):

                domain.enter_case(i)

                L_true = [0]*node_sat.get_number_of_states()
                L_true[node_sat.get_case_state(i)] = 1
                L_true_total.append(L_true)
                
                node_sat.retract_findings()
                domain.propagate()

                L_predict = ([node_sat.get_belief(oi) for oi in range(node_sat.get_number_of_states())])
                L_predict_total.append(L_predict)
                
                domain.initialize()
             score = brier_multi(L_true_total, L_predict_total)

        #Delete domain and data for memory
        domain.delete()
        data.delete()
    
    except HuginException:
        print("A Hugin Exception was raised!")
        raise
    return nb_step

        

    










#Not used function

def fine_tuning_missing_AUC(name_model, data_name, L_ini, eta, beta, alpha):

    try:
  
        data = DataSet.parse_data_set(
            data_name, separator=',', error_handler=None)

        domain = Domain.parse_domain(name_model)
        L_nodes = domain.get_nodes()

        L_auc = []
        AUC = 0
        ER = 0

        node_sat = domain.get_node_by_name(name_class)
        nb_class_node_sat = node_sat.get_number_of_states()

        # Initialize the CPT randomly (same initialization for the different algorithm)
        for i in range(len(L_nodes)):
            L_nodes[i].get_table().set_data(L_ini[i])


        domain.add_cases(data, 0, data.get_number_of_rows())

        domain.compile()

        #Initialize experience table
        for i in L_nodes:
            size_table = i.get_experience_table().get_size()
            i.get_experience_table().set_data([1/size_table]*size_table)


        domain.learn_tables()


        domain.set_number_of_cases(0)

        domain.add_cases(data, 0, data.get_number_of_rows())

        L_nodes = node_sat.get_children()



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
                   
             
            ER_l=roc_auc_score(L_true,L_predict,multi_class="ovo",average="weighted")     
        else:
            for i in range(0,domain.get_number_of_cases()):
                 
                 domain.enter_case(i)
         
                 L_true.append(node_sat.get_case_state(i))       
                 
                 node_sat.retract_findings()
                 domain.propagate()
                 L_predict.append(node_sat.get_belief(1))
         
                 domain.initialize()      
         
           
   
            ER_l=roc_auc_score(L_true,L_predict)

        max_ER = 2
        nb_step=-1
        while(ER_l < max_ER): 
        
            domain.save_as_net(BFTNB_model)
            max_ER = ER_l
            nb_step+=1        
            for i in range(0, domain.get_number_of_cases()):

                domain.enter_case(i)

                c_actuall = node_sat.get_case_state(i)
                node_sat.retract_findings()
                domain.propagate()
                c_predict = np.argmax([node_sat.get_belief(
                    jk) for jk in range(node_sat.get_number_of_states())])
                domain.initialize()

                if c_predict != c_actuall:
                    error = abs(node_sat.get_belief(c_actuall) - node_sat.get_belief(c_predict))

                    for node in L_nodes:
                        if node.case_is_set(i):  
                            
                            ai = node.get_case_state(i)                        
                            table = creat_table(node, i, c_actuall, c_predict, 0.001, alpha, beta, error, nb_class_node_sat, ai)
                            node.get_table().set_data(table, 0, len(table))


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
                       
                 
                ER_l=roc_auc_score(L_true,L_predict,multi_class="ovo",average="weighted")     
            else:
                for i in range(0,domain.get_number_of_cases()):
                     
                     domain.enter_case(i)
             
                     L_true.append(node_sat.get_case_state(i))       
                     
                     node_sat.retract_findings()
                     domain.propagate()
                     L_predict.append(node_sat.get_belief(1))
             
                     domain.initialize()      
             
               
   
                ER_l=roc_auc_score(L_true,L_predict)


        domain.delete()
        data.delete()
    
    except HuginException:
        print("A Hugin Exception was raised!")
        raise
    return nb_step
