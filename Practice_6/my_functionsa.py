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
    return pd.read_csv(csv_file), np.load(npy_file, allow_pickle=True)


def compute_biometric_scores(people, people_embeddings,
                              connections, connection_embeddings,
                              mode='only_compare_same_logins', 
                              distance_fct=np.linalg.norm):
    
    if mode=='only_compare_same_logins':
        distances = []
        for idx, connection in connections.iterrows():
            target_login = connection['logins']
            target_person = people[people['logins']==target_login]
            index_of_target_person = target_person.index[0]
            target_person_emb = people_embeddings[index_of_target_person]
            connection_emb = connection_embeddings[idx]
            distance = distance_fct(target_person_emb - connection_emb)
            distances.append(distance)
        
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
    
    fig, ax = plt.subplots(figsize=(10,10))

    ax.hist(true_scores,  label='True Scores',  bins=100, alpha=0.5)
    ax.hist(false_scores, label='False Scores', bins=100, alpha=0.9)

    ax.set_title("Scores of good and bad connections")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Amount of scores")
    ax.legend()
    
    if bool(save_figure):
        plt.savefig(save_figure)
    else:
        plt.show()
    
    
def ROC_curve(FAR, TAR,
              save_figure=False):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(FAR, TAR)
    ax.set_title("ROC curve")
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR")
    if bool(save_figure):
        plt.savefig(save_figure)
    else:
        plt.show()
    
    
def who_is_in(people, connections, threshold):
    """ 
    inputs:
    [required] people: pd.DataFrame of registered people, containing logins, passwords
    [required] connections: pd.DataFrame of authentication attempts, containing logins, passwords, biometric scores
    Computes which connections are accepted or rejected.
    output:
    accepted: list of booleans, same length as connections, True if the connection attempts has been accepted.
    """
    accepted=[]
    #print("You have to code this function.")
    
    for idx, connection in connections.iterrows(): 
        accepted.append(connection['scores']<threshold)
                        
    assert len(accepted)==len(connections) #making sure this is the right length
    return accepted
    
    
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
    TA = np.sum([A and B for A,B in zip(accepted_list, true_list)])
    FA = np.sum([A and not B for A,B in zip(accepted_list, true_list)])
    FR = np.sum([not A and B for A,B in zip(accepted_list, true_list)])
    TR = np.sum([not A and not B for A,B in zip(accepted_list, true_list)])
    Metrics['TAR'] = 100*TA/(TA+FA)
    Metrics['FAR'] = 100*FA/(TA+FA)
    Metrics['TRR'] = 100*TR/(TR+FR)
    Metrics['FRR'] = 100*FR/(TR+FR)
    Metrics['Accuracy'] = 100*(TA+TR)/(TA+TR+FR+FA)
    
    
    
    
    output = ""
    for key in Metrics:
        output+= f"{key}: {Metrics[key]}\n"
    if print_output:
        print(output)
    if save_output:
        with open(log_file, 'w+') as f:
            f.writelines(output)

    # return best res
    return Metrics['TAR'], Metrics['FAR']
    

    
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
    