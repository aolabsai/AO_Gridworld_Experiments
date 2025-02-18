


import ao_arch as ar
import numpy as np


number_qa_neurons = 20

description = "Basic Clam"
arch_i = [1, 1, 1, 1, 1, 1]     # 3 neurons, 1 in each of 3 channels, corresponding to Food, Chemical-A, Chemical-B (present=1/not=0)
arch_z = [2]           # corresponding to Open=1/Close=0
arch_c = [0]           # adding 1 control neuron which we'll define with the instinct control function below
arch_qa = [number_qa_neurons]

connector_function = "full_conn"
pain_signal = False


# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, arch_qa=arch_qa, qa_conn="full", description=description)


#Adding Aux Action
def qa0_firing_rule(INPUT, Agent): 
    pain_signal = False
    from ao_agent import reset_qa
    if reset_qa:
        print("reset of qa")
        Agent.counter = number_qa_neurons
    if not hasattr(Agent, 'counter'):
        Agent.__setattr__("counter", 20)

    if Agent.counter == 0:
        pain_signal = True
        print("pain signal true")
        group_response = np.ones(number_qa_neurons)

    elif Agent.counter < (number_qa_neurons+1):
        Agent.counter -= 1
        group_response = np.zeros(number_qa_neurons)
        group_response[0 : Agent.counter] = 1

            

    else:    #If the agent did not react then dont touch the counter
        group_response = np.zeros(number_qa_neurons)
        group_response[0 : Agent.counter] = 1

     

    group_meta = np.ones(number_qa_neurons, dtype="O")
    group_meta[:] = "qa0"
    return group_response, group_meta
# Saving the function to the Arch so the Agent can access it
Arch.datamatrix_aux[2] = qa0_firing_rule

