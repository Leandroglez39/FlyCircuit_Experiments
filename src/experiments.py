from matrix import *
import re
import matplotlib as ptl
import pandas as pd
import math



def read_GT(path: str) -> dict:

    '''
    Read ground truth from file

    Parameters
    ----------
    path: str
        Path to folder containing ground truth files

    Returns
    -------
    gt: dict
        Ground truth dict for all networks in the folder. The key is the network name, 
            the value is a list of ground truth communities
    '''

    gt = {}

    files = os.listdir(path)
    
    for file in files:

        with open(path + '/' + file, 'r') as f:
            lines = f.readlines()
            networtk_gt = []
            for line in lines:
                communities = [int(node) for node in line.split(' ')]               
                networtk_gt.append(communities)
            gt[file] = networtk_gt
    
    return gt

def read_rc_output(path: str) -> dict:

    '''
    Read communities from file

    Parameters
    ----------
    path: str
        Path to folder containing community files

    Returns
    -------
    communities: dict
        Communities dict for all networks in the folder. The key is the network name, 
            the value is a list of communities
    '''

    communities = {}

    files = os.listdir(path)
    
    for file in files:

        with open(path + '/' + file, 'r') as f:
            lines = f.readlines()
            clean_lines = []
            for line in lines:
                clean_lines.append(line.strip('\n').rstrip())
            lines = clean_lines
            networtk_communities = []
            for line in lines:
                communitie = [int(node) for node in line.split(' ')]               
                networtk_communities.append(communitie)
            communities[file] = networtk_communities
    
    return communities

def overlaping_detection(dict: dict) -> dict:
    
        '''
        Detect overlaping communities
    
        Parameters
        ----------
        dict: dict
            Communities dict for all networks in the folder. The key is the network name, 
                the value is a list of communities
    
        Returns
        -------
        overlaping: dict
            Overlaping communities dict for all networks. The key is the network name, 
                the value is a set of overlaping nodes.
        '''
    
        overlaping = {}
    
        for key, value in dict.items():
            nodes = set()
            overlaping[key] = set()
            for community in value:
                intersection = nodes.intersection(set(community))
                if len(intersection) > 0:
                    overlaping[key].update(intersection)
                    nodes.update(set(community))
                else:
                    nodes.update(set(community))
        
        return overlaping

def generate_pkl(path: str) -> None:

    '''
    Generate pkl files from txt files

    Parameters
    ----------
    path: str
        Path to folder containing txt files

    Returns
    -------
    None
    '''

    files = os.listdir('dataset/' + path)
    files.remove('GT')
    files.remove('README.txt')

    for file in files:
        G = nx.Graph()
        with open('dataset/' + path + '/' + file + '/' + file + '.dat', 'r') as f:
            lines = f.readlines()
                
        for line in lines:
            a, b = line.split('\t')
            G.add_edge(int(a) - 1, int(b) - 1)

        #     #nx.nx_pylab.draw(G, with_labels=True)
        #     #plt.show()
        pickle.dump(G, open('dataset/' + path + '/' + file + '/' + file + '.pkl', 'wb'))
        G.clear()

def runAlgorithmSimple(m, folder_version = 'NetsType_1.3'):

    for j in range(1, 12):

        m.G = pickle.load(open('dataset/' + folder_version + '/network'+ str(j) + '/network'+ str(j) + '.pkl', 'rb'))

        n = 0
        top = 1

        exportpath_Simple = folder_version

        for i in range(n, top):
            result = nx.algorithms.community.label_propagation.asyn_lpa_communities(m.G, seed=random.randint(0, 10000))
            communities = [list(x) for x in result]
            m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Lpa.txt', communities)
        
        for i in range(n, int(top)):
            result = nx.algorithms.community.greedy_modularity_communities(m.G, resolution= 1)        
            communities = [list(x) for x in result]
            m.export_Simple(exportpath_Simple, '/network'+ str(j) +'_Greedy.txt', communities)
        
        for i in range(n, top):
            result = nx.algorithms.community.louvain.louvain_communities(m.G, seed=random.randint(0, 10000))
            communities = [list(x) for x in result]
            m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Louvain.txt', communities)

    print('done')


def runAlgorithmSimpleTunning(m, init, step, top, folder_version = 'NetsType_1.1_Tunning'):

    exportpath_Simple = folder_version
    folder_version = folder_version.split('_Tunning')[0]

    for j in range(1, 12):

        m.G = pickle.load(open('dataset/' + folder_version + '/network'+ str(j) + '/network'+ str(j) + '.pkl', 'rb'))

        size = (top - init)/step
        resolution = [init+(step*i) for i in range(int(size)+1)]


        # for i in range(n, top):
        #     seed_i = random.randint(init, end)
        #     result = nx.algorithms.community.label_propagation.asyn_lpa_communities(m.G, seed=seed_i)
        #     communities = [list(x) for x in result]
        #     m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Lpa_' + str(i)  + '.txt', communities)

        
        for rs_i in resolution:
            result = nx.algorithms.community.greedy_modularity_communities(m.G, resolution = rs_i)        
            communities = [list(x) for x in result]
            m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Greedy_' + str(rs_i)  + '.txt', communities)
        
        
        # for rs_i in resolution:
        #     result = nx.algorithms.community.louvain.louvain_communities(m.G, resolution = rs_i, seed=random.randint(0, 10000))
        #     communities = [list(x) for x in result]
        #     m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Louvain_' + str(rs_i)  + '.txt', communities)

    print('done')


def runAlgorithmSimpleTunning(m, init, step, top, folder_version = 'NetsType_1.1_Tunning'):

    exportpath_Simple = folder_version
    folder_version = folder_version.split('_Tunning')[0]

    for j in range(1, 12):

        m.G = pickle.load(open('dataset/' + folder_version + '/network'+ str(j) + '/network'+ str(j) + '.pkl', 'rb'))

        size = (top - init)/step
        resolution = [init+(step*i) for i in range(int(size)+1)]


        # for i in range(n, top):
        #     seed_i = random.randint(init, end)
        #     result = nx.algorithms.community.label_propagation.asyn_lpa_communities(m.G, seed=seed_i)
        #     communities = [list(x) for x in result]
        #     m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Lpa_' + str(i)  + '.txt', communities)

        
        for rs_i in resolution:
            result = nx.algorithms.community.greedy_modularity_communities(m.G, resolution = rs_i)        
            communities = [list(x) for x in result]
            m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Greedy_' + str(rs_i)  + '.txt', communities)
        
        
        # for rs_i in resolution:
        #     result = nx.algorithms.community.louvain.louvain_communities(m.G, resolution = rs_i, seed=random.randint(0, 10000))
        #     communities = [list(x) for x in result]
        #     m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Louvain_' + str(rs_i)  + '.txt', communities)

    print('done')

def drawResultAlgorithm(folderpath, nameFile, ga = '0.8'):

    print('Begin!!!!!!!!!!!')

    dictResult = pickle.load(open('output/' + folderpath + '/' + nameFile, 'rb'))
    dictResult[f'RC']['Algorithms/Parameters'] = f'RC_{0.8}'

    gammas = [ '0.5', '0.6' , '0.7']

    for gamma in gammas:

        dictResult2 = pickle.load(open('output/' + 'gamma_' + gamma + '/' + folderpath + '/' + nameFile, 'rb'))
    
        dictResult[f'RC_{gamma}'] = dictResult2['RC']
        dictResult[f'RC_{gamma}']['Algorithms/Parameters'] = f'RC_{gamma}'
    
    #dictResult['RC'] = dictResult2['RC']
    #dictResult[f'RC_{gamma}'] = dictResult2['RC']

    df = pd.DataFrame()

    for _ , v in dictResult.items():
        df = df.append(v, ignore_index = True)
    
    columnsSorted = sorted(df.columns)
    nameColumns = columnsSorted[0]
    columnsSorted.remove(nameColumns)
    columnsSorted = sorted(columnsSorted, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    columnsSorted.insert(0, nameColumns)
    
    df = df.reindex(columnsSorted, axis=1)
    df = df.set_index(nameColumns)
    
    print('created df done')
    #print(df.columns)
    for i in range(1, 12):
        df = df.rename(columns={f'network{i}': float(f'{(i-1)*0.05 + 0.1}')})
    
    
    #print(df)
    dfT = df.transpose()
    #print(dfT)
    
    # Sort dataframe by column names
    dfT = dfT.reindex(sorted(dfT.columns), axis=1)

    print('df transpose done')
    markers = ['o', 's', '^', 'p', '*', '+', 'x', 'D']
    index = 0
    for item in dfT.columns:
        dfT[item].plot(kind='line', marker=markers[index], label=item)
        index += 1
        
           
       
    #plt.title('Run Algorithms and NMI accuracy' + ' Network: ' + folderpath)
    plt.xlabel('Parameter µ in LFR Benchmark') 
    plt.ylabel('NMI Values')
    plt.xticks(np.arange(0.1, 0.65, step=0.05))
    plt.legend()
    #plt.show()
    
    plt.savefig(f'output/{folderpath}/' + folderpath + '.png', dpi=700)
  

def drawStability2(folder_version: str):

    df = pd.read_csv(f'output/stability/{folder_version}/nmi_stability.csv', header=0)

    nets = df['Network'].unique()

    df = df.groupby(['Network', 'Algorithm', 'Iterations']).mean().reset_index()

    if folder_version == 'NetsType_1.4':
        df.loc[df['Iterations'] == 10, 'Iterations'] = 1
        df.loc[df['Iterations'] == 100, 'Iterations'] = 2
        df.loc[df['Iterations'] == 1000, 'Iterations'] = 3
        labels = ["10", "100", '1000']
    else:
        df.loc[df['Iterations'] == 10, 'Iterations'] = 1
        df.loc[df['Iterations'] == 100, 'Iterations'] = 3
        df.loc[df['Iterations'] == 50, 'Iterations'] = 2
        labels = ["10", "50", '100']

 

    
    
    for network in df['Network'].unique():
        network_data = df[df['Network'] == network]
        markers = ['o', 's', '^', 'p', '*'] # list of markers to use
        for marker, algorithm in zip(markers, network_data['Algorithm'].unique()):
            algorithm_data = network_data[network_data['Algorithm'] == algorithm]
            plt.style.use('seaborn-v0_8-darkgrid')
            plt.plot(algorithm_data['Iterations'], algorithm_data['NMI'], label=algorithm, marker=marker, linestyle='dotted', markersize=10)
        plt.title(f'Stability in {network}')
        plt.xlabel('Number of runs')
        plt.ylabel('NMI Average')
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xticks([1,2,3], labels)
        plt.legend()
        plt.savefig(f'output/stability/{folder_version}/nmi_img/{network}.png', dpi=550)
        plt.clf()
    
        

    # df = df[df['Network'] == 'network10']
    # df1 = df[df['Algorithm'] == 'async_lpa']
    # plt.scatter(df1['Iterations'], df1['NMI'], label='async_lpa', color='blue', marker='o')  # type: ignore
    # df2 = df[df['Algorithm'] == 'RC']
    # plt.scatter(df2['Iterations'], df2['NMI'], label='RC', color='red', marker='s') # type: ignore
    # plt.title(f'Run Algorithms and NMI accuracy in network1 NetsType_1.4')
    # plt.xlabel('Iterations')
    # plt.ylabel('NMI Accuracy')
    # plt.yticks(np.arange(0, 1.2, 0.2))
    # plt.xticks([1,2,3], labels)
    # #plt.legend()
    # plt.show()

    # for network in df['Network'].unique():
    #     network_data = df[df['Network'] == network]
    #     for algorithm in network_data['Algorithm'].unique():
    #         algorithm_data = network_data[network_data['Algorithm'] == algorithm]
    #         plt.boxplot([algorithm_data[algorithm_data['Iterations'] == label]['NMI'] for label in labels], labels=labels, patch_artist=True)
    #     plt.title(f'Run Algorithms and NMI accuracy in {network} NetsType_1.4')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('NMI Accuracy')
    #     #plt.legend()
    #     plt.show()
   
if __name__ == "__main__":

    # create Data Structure
    m = Matrix([], {},[])
    # run Algorithm simple
    #runAlgorithmSimpleTunning(m, 5.5, 0.5, 10.0, 'NetsType_1.1_Tunning')
    # draw result
    drawResultAlgorithm('NetsType_1.4', 'NetsType_1.4_result.pkl')
    
    #drawStability2('NetsType_1.4')

    
    #drawStability2('NetsType_1.6')

    #G = pickle.load(open('dataset/NetsType_1.6/network10/network10.pkl', 'rb'))

    # result = nx.algorithms.community.louvain.louvain_communities(G, seed=random.randint(0, 10000), resolution=random.uniform(2,3.5)) # type: ignore

    # communities = [list(x) for x in result] # type: ignore

    # pickle.dump(communities, open('dataset/NetsType_1.6/network10/network10_Louvain.pkl', 'wb'))

    # from cdlib import evaluation, NodeClustering

    # communities = pickle.load(open('dataset/NetsType_1.6/network10/network10_Louvain.pkl', 'rb'))



    # nodes= []

    # with open('dataset/' + 'NetsType_1.6' + '/GT/community' + '10' + '_GT.dat', 'r') as f:
    #             lines = f.readlines()        
    #             for line in lines:
    #                 data = line.split(' ')
    #                 inter_data = [int(x) for x in data]
    #                 nodes.append(inter_data)

    # nodeClustA = NodeClustering(communities=nodes, graph=G, method_name='GT', method_parameters={}, overlap=True)
    # nodeClustB = NodeClustering(communities=communities, graph=G, method_name='Louvain', method_parameters={}, overlap=True)

    # match_resoult = evaluation.overlapping_normalized_mutual_information_MGH(nodeClustA, nodeClustB)

    # print(match_resoult)


