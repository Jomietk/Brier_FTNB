

import os

os.chdir('C://Users//jomie//.spyder-py3//Fine_tuning//Fine_tuning_NB_NA_conflict')



import numpy as np
import pandas as pd


import sys
from pyhugin91 import *





    #%%
    
    

name_class="class"

def model_creation(name_data2,name_model):
    
    
    data_=pd.read_csv(name_data2)

    domain = None

    #CL=ClassCollection()
    # create domain
    #BNclass= Class(CL)
    
    domain =Domain()
    data=DataSet.parse_data_set(name_data2, separator=',')
    
    for i in range(data.get_number_of_columns()):
        name=data.get_column_name(i)
        col=data_[name]
# =============================================================================
#         c=0
#         while str(col[c])=="nan":
#             c=c+1
# =============================================================================
        St=list(set(col))            
        St=list({x for x in St if x==x})
        o=St[0]
        if len(St)>1:
            if isinstance(o, str):     
                node = Node(domain, CATEGORY.CHANCE, KIND.DISCRETE, SUBTYPE.LABEL)
                
            
                
                node.set_name(name)
                node.set_label(name)
                St=list(set(data_[name]))            
                St=list({x for x in St if x==x})
            
                node.set_number_of_states(len(St))
     
                for p in range(len(St)):
    
                    node.set_state_label(p,St[p])
                    
                 
                    
                    
            else:
                node = Node(domain, CATEGORY.CHANCE, KIND.DISCRETE, SUBTYPE.NUMBER)
                node.set_name(name)
                node.set_label(name)
                St=np.sort(list(set(data_[name])))
           
                St=list({x for x in St if x==x})
              
                node.set_number_of_states(len(St))
             
             
                for p in range(len(St)):
                    node.set_state_value(p,St[p])
                
                

    
    #print(data.get_column_name(0))
    #domain.parse_cases("Ratata.csv", error_handler=None)
    
    node_class=domain.get_node_by_name(name_class)
    for i in domain.get_nodes():
        if i.get_name()!=name_class:
            i.add_parent(node_class)
        
   
   
    domain.compile()
    
 
    domain.save_as_net(name_model) 
    domain.delete()


