from matrix import *
import re

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


   
if __name__ == "__main__":

    G = pickle.load(open('dataset/NetsType_1.1/network8/network8.pkl', 'rb'))

    result = nx.algorithms.community.label_propagation.asyn_lpa_communities(G, seed=random.randint(0, 10000))
    communities = [list(x) for x in result]
    print(len(communities))


    

    #generate_pkl('NetsType_1.1')

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