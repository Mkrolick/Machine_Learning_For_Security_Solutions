import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics


def load_files(csv_file="",
               npy_file=""):
    """
    [required]csv_file - str: path to the csv file to be loaded with pandas
    [required]npy_file - str: path to the npy file to be loaded with numpy
    returns: a pd.DataFrame and a numpy array
    """
    print("This function needs to be coded")
    return 


def compute_biometric_scores(people, people_embeddings,
                              connections, connection_embeddings,
                              mode='only_compare_same_logins', 
                              distance_fct=np.linalg.norm):
    
    if mode=='only_compare_same_logins':
        distances = []
        
        print("This function needs to be coded")
        
        return distances
    else:
        raise ValueError(f"mode specified unknown. Accepted modes='only_compare_same_logins'. You proposed: {mode}")
        

def scatter_plot(X,Y,
                 labels=[],
                 save_figure=False):
    if len(labels)==0: labels = np.zeros(len(X))#if no labels are provided, same labels for everyone
    
    fig, ax = plt.subplots(figsize=(10,10))
    for i in set(labels):
        ax.scatter(X[labels==i], Y[labels==i], label=i)
        
    #ax.legend()
    ax.set_title("Scatter Plot of Embeddings")
    ax.set_xlabel("Emb dimension 1")
    ax.set_ylabel("Emb dimension 2")
    
    
    if bool(save_figure):
        plt.savefig(save_figure)
    else:
        plt.show()
    

def scores_histogram(true_scores, false_scores,
                     save_figure=False):
    
    print("This function needs to be coded")
    
    if bool(save_figure):
        plt.savefig(save_figure)
    else:
        plt.show()
    
    
def ROC_curve(FAR, TAR,
              save_figure=False):
    fig, ax = plt.subplots(figsize=(10,10))
    
    print("This function needs to be coded")
    
    if bool(save_figure):
        plt.savefig(save_figure)
    else:
        plt.show()
    
    
def who_is_in(people, connections, theshold):
    """ 
    inputs:
    [required] people: pd.DataFrame of registered people, containing logins, passwords
    [required] connections: pd.DataFrame of authentication attempts, containing logins, passwords, biometric scores
    Computes which connections are accepted or rejected.
    output:
    accepted: list of booleans, same length as connections, True if the connection attempts has been accepted.
    """
    
    accepted=[]
    
    print("This function needs to be coded")
    
    return 
    
    
def get_all_metrics(accepted_list, 
                    true_list,
                    print_output=True,
                    save_output=True,
                    log_file='output_default.log'):
    """ 
    inputs:
    [required] accepted_list: list of booleans for each connection: where they accepted?
    [required] true_list: list of booleans for each connection: where they supposed to be accepted?
    Computes a set of metrics, print it and/or save it in a file.
    """
    Metrics = {}
    Metrics['TAR'] = 100*np.sum([A and B for A,B in zip(accepted_list, true_list)])/np.sum(accepted_list)
    Metrics['FAR'] = 100*np.sum([A and not B for A,B in zip(accepted_list, true_list)])/np.sum(accepted_list)
    Metrics['TRR'] = 100-Metrics['FAR']
    Metrics['FRR'] = 100-Metrics['TAR']
    Metrics['Accuracy'] = (Metrics['TAR']+Metrics['TRR']) / 2
    
    
    
    
    output = ""
    for key in Metrics:
        output+= f"{key}: {Metrics[key]}\n"
    if print_output:
        print(output)
    if save_output:
        with open(log_file, 'w+') as f:
            f.writelines(output)
    

    
if __name__=="__main__":
    #if you launch this file, it is going to execute the following code.
    #If you import the functions from here, it's not going to execute the following code.
    
    #### LOADING DATA ####
    people, people_embeddings = load_files(csv_file="people.csv", npy_file="people_embeddings.npy")
    connections, connection_embeddings = load_files(csv_file="connections.csv", npy_file="connections_embeddings.npy")
    
    #### ANALYSIS OF DATA ####
    #plot the embeddings of true attempts
    true_attempts = connections[connections['true_attempt']==True]
    embeddings = connection_embeddings[list(true_attempts.index)]
    scatter_plot(X=embeddings[:,0],
                 Y=embeddings[:,1],
                 labels=np.array(true_attempts['logins']),
                 save_figure='scatterplot.png'
                )
    
    #compute the scores
    connections['scores'] = compute_biometric_scores(people, people_embeddings, 
                                                      connections, connection_embeddings,
                                                      mode='only_compare_same_logins', 
                                                      distance_fct=np.linalg.norm)
    
    #plot the histogram of scores
    scores_histogram(list(connections[connections['true_attempt']==True]['scores']),
                     list(connections[connections['true_attempt']==False]['scores']),
                     save_figure="histogram_of_my_scores.png")
    
    
    #get FAR and TAR using sklearn function
    TAR, FAR, thresholds = metrics.roc_curve(list(connections['true_attempt'].astype(int)),
                                            list(connections['scores'])
                                           )
    # plot the ROC curve
    ROC_curve(FAR, TAR, 'my_roc_curve.png') 
    
    
    #### AUTHENTICATION SYSTEM ####
    # Find who should get in the system
    accepted = who_is_in(people, connections)
    connections['accepted']=accepted
    
    # Measure how is your system behaving
    get_all_metrics(accepted, 
                    list(connections['true_attempt']), 
                    print_output=True,
                    save_output=True,
                    log_file='output_week5.log')
    