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
    
    #print("This function needs to be coded")
    return pd.read_csv(csv_file), np.load(npy_file)


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
            
            distance = distance_fct(target_person_emb-connection_emb)
            
            distances.append(distance)

        connections['distance']=distances
        connections.head()
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

#scores_histogram(list(connections[connections['true_attempt']==True]['scores']),
#                 list(connections[connections['true_attempt']==False]['scores']),
#                 save_figure=False)

    true_list_distances  = true_scores
    false_list_distances = false_scores

    ax.hist(true_list_distances,  label="true_scores"     ,  bins=100, alpha=0.5)
    ax.hist(false_list_distances,  label="false_scores" , bins=100, alpha=0.5)

    ax.set_title("Scores of good and bad connections")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Amount of scores")
    ax.legend()
    plt.show()
    
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
    plt.show()
    
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

    # iterate over the connections pd.DataFrame
    for index, row in connections.iterrows():
        # get the login and password of the connection attempt
        login = row['logins']
        password = row['passwords']
        # get the biometric score of the connection attempt
        score = row['scores']
        # get the index of the person in the people pd.DataFrame
        person_index = people[people['logins']==login].index[0]
        # get the password of the person
        person_password = people.loc[person_index]['passwords']

        # get the biometric score of the person
        # check if the password is correct

        ip_connection = row['ips']
        ip_person = people.loc[person_index]['ips']
        
        email_connection = row['logins']
        email_person = people.loc[person_index]['logins']

        if ip_connection == ip_person and email_connection == email_person and password == person_password:
            accepted.append(True)
        else:
            accepted.append(False)
    
    return accepted

        


    
        
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
    