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


def runAlgorithmSimpleTunning(m, folder_version = 'NetsType_1.1_Tunning'):

    exportpath_Simple = folder_version
    folder_version = folder_version.split('_Tunning')[0]

    for j in range(1, 12):

        m.G = pickle.load(open('dataset/' + folder_version + '/network'+ str(j) + '/network'+ str(j) + '.pkl', 'rb'))

        n = 1
        top = 5
        
        louvainParameters = []
        greedyParameters = []

        

        init = 0
        end = 250
        lpaParameters = []

        # for i in range(n, top):
        #     seed_i = random.randint(init, end)
        #     result = nx.algorithms.community.label_propagation.asyn_lpa_communities(m.G, seed=seed_i)
        #     communities = [list(x) for x in result]
        #     m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Lpa_' + str(i)  + '.txt', communities)
        #     init = end
        #     end = end + 250
        
        resolution = [3.5, 4.0, 4.5, 5.0]
        for i in range(n, int(top)):
            rs = resolution[i-1]
            result = nx.algorithms.community.greedy_modularity_communities(m.G, resolution= rs)        
            communities = [list(x) for x in result]
            m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Greedy_' + str(i)  + '.txt', communities)
        
        # resolution = [3.5, 4.0, 4.5, 5.0]
        # for i in range(n, top):
        #     rs = resolution[i-1]
        #     result = nx.algorithms.community.louvain.louvain_communities(m.G, resolution = rs, seed=random.randint(0, 10000))
        #     communities = [list(x) for x in result]
        #     m.export_Simple(exportpath_Simple, '/network'+ str(j) + '_Louvain_' + str(i)  + '.txt', communities)

    print('done')

def drawResultAlgorithm(folderpath, nameFile):

    print('Begin!!!!!!!!!!!')

    columnsName = ['Algorithms', 'network1', 'network2', 'network3', 'network4', 'network5',
               'network6', 'network7', 'network8', 'network9', 'network10', 'network11']
    # algName = ['RC', 'Lpa', 'Louvain', 'Greedy', '_Infomap']
    algName = ['Greedy_1', 'Greedy_2', 'Greedy_3', 'Greedy_4']
    df = pd.DataFrame(columns= columnsName)
    df['Algorithms'] = algName

    with open('output/' + folderpath + '/' + nameFile, 'r') as file:
        row_i =[]
        currentLine = file.readline().replace('\n', '')
        while currentLine:
            if 'network' in currentLine:
                net = currentLine
                currentLine = file.readline().replace('\n', '')
                scoresList = []
                while '---------' not in currentLine:
                    score_i = currentLine.split(': ')
                    sc = round(float(score_i[1]), 2)
                    scoresList.append(sc)
                    currentLine = file.readline().replace('\n', '')
                df[net] = scoresList
            currentLine = file.readline().replace('\n', '')
            
    print('created df done')
    print(df)

    allScores = []
    scRC = []
    scLpa = []
    scLouvain = []
    scGreedy = []
    # scInfomap = []
    for i in range(len(df)):
        row_iValues = [df.iloc[i,j] for j in range(1,12)]
        allScores.append(row_iValues)

    print('begin draw')

    scRC = allScores[0]
    scLpa = allScores[1]
    scLouvain = allScores[2]
    scGreedy = allScores[3]
    # scInfomap = allScores[4]



    nets = range(1,12)
    plt.plot(nets, scRC, 'r', label= str(algName[0]) + ' resolution: 3.5 ' + ' accuracy')
    plt.plot(nets, scLpa, 'g', label= str(algName[1]) + ' resolution: 4.0 ' + ' accuracy')
    plt.plot(nets, scLouvain, 'c', label= str(algName[2]) + ' resolution: 4.5 ' + ' accuracy')
    plt.plot(nets, scGreedy, 'm', label= str(algName[3]) + ' resolution: 5.0 ' + ' accuracy')
    # plt.plot(nets, scRC, 'r', label='RC accuracy')
    # plt.plot(nets, scLpa, 'g', label='Lpa accuracy')
    # plt.plot(nets, scLouvain, 'c', label='Louvain accuracy')
    # plt.plot(nets, scGreedy, 'm', label='Greedy accuracy')
    # plt.plot(nets, scInfomap, 'b', label='Infomap accuracy')
    
    plt.title('Run Algorithms and NMI accuracy' + ' Network: ' + folderpath)
    plt.xlabel('Nets')
    plt.ylabel('NMI Accuracy')
    plt.legend()
    plt.show()

   
if __name__ == "__main__":

    # create Data Structure
    m = Matrix([], {},[])
    # run algorithm
    # runAlgorithmSimple(m, 'NetsType_1.1')
    # runAlgorithmSimpleTunning(m, 'NetsType_1.1_Tunning')
    drawResultAlgorithm('NetsType_1.1_Tunning', 'NetsType_1.1_Tunning_result.txt')

    #generate_pkl('NetsType_1.3')

    # G = pickle.load(open('dataset/NetsType_1.1/network8/network8.pkl', 'rb'))

    # result = nx.algorithms.community.label_propagation.asyn_lpa_communities(G, seed=random.randint(0, 10000))
    # communities = [list(x) for x in result]
    # print(len(communities))


    

    

    # gt = read_GT('LFRBenchamark')
    # rc = read_rc_output('output/rc')

    # over_gt = overlaping_detection(gt)
    # over_rc = overlaping_detection(rc)


    # for key, value in over_gt.items():
    #     match = re.search(r'(community)(\d+)(_GT.dat)', key)
    #     if match:
    #         number = match.group(2)
    #         rc_filename = 'network' + number + '_result.dat'
    #         overlapint_iteration = over_rc[rc_filename]
    #         print(rc_filename, overlapint_iteration)
    #         print(len(value.intersection(overlapint_iteration)))