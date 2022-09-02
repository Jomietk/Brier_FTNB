



import os

os.chdir('C://Users//jomie//.spyder-py3//Fine_tuning//Fine_tuning_NB_NA_conflict')




def creat_table(node,i,c_actuall,c_predict,eta,alpha,beta,error,nb_class_node_sat,ai):
    
        
    table=node.get_table().get_data()
    nb_states=node.get_number_of_states()
    
    p_actuall=table[c_actuall*nb_states+ai]
    p_max=max([table[c_actuall*nb_states+jk] for jk in range(nb_states)])  
    p_actuall= p_actuall+eta*(alpha*p_max-p_actuall)*error
    
    table[c_actuall*nb_states+ai]=p_actuall
    
     
    p_predict=table[c_predict*nb_states+ai]  
    p_min=min([table[c_predict*nb_states+jk] for jk in range(nb_states)])     
    if p_predict-eta*(beta*p_predict-p_min)*error>0:
        p_predict= p_predict-eta*(beta*p_predict-p_min)*error
    
    table[c_predict*nb_states+ai]=p_predict

    return table








