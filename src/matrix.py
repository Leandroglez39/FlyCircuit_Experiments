import time
import pandas as pd
import pickle
from dataclasses import dataclass
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import multiprocessing
import os
import datetime
import math




@dataclass
class Matrix:
    """    
    Class to create a matrix from a csv file and export it to a graphml file    
    
    Attributes:
        list_nodes (list): list of nodes
        pos_code_nodes (dict): dictionary with the position of each node
        G (nx.DiGraph): graph
        ady_list (list): list of adyacents nodes
        G_small (nx.DiGraph): small graph
    """

    list_nodes: list
    pos_code_nodes: dict
    G = nx.DiGraph()
    ady_list : list 
    G_small = nx.DiGraph()
    
    def __post_init__(self):        
        self.list_nodes = pickle.load(open('./dataset/nodes.pkl', 'rb'))
        self.pos_code_nodes = pickle.load(open('./dataset/pos_code_nodes.pkl', 'rb'))
        self.ady_list = [[] for _ in range(len(self.list_nodes))]
        

    def insert_nodes(self):
        self.G.add_nodes_from(self.list_nodes)
    
    def insert_weighted_edges(self):
        for i in range(len(self.ady_list)):
            for j in range(len(self.ady_list[i])):
                y = self.ady_list[i][j][0]
                weight = self.ady_list[i][j][1]
                self.G.add_edge(self.list_nodes[y], self.list_nodes[i], weight=weight)
    
    def export_graph_to_csv(self, path = "./dataset/graph_19k_3.5m.csv"):
        '''
        Export graph to csv file for Cosmos viewer
        '''
        nx.write_edgelist(self.G, path, delimiter=",", data=['weight'])

    def export_graph_to_csv_size(self, size):

        G = nx.DiGraph()
        G.add_nodes_from(self.list_nodes[:size])

        df = pd.read_csv('./dataset/matrix/0 file.csv')
        

        ady_list = [[] for _ in range(size)]

        for i in range(len(self.list_nodes[:size])):
            row = df.loc[i].to_list()
            for j in range(size):
                if row[j] != 0:
                    ady_list[i].append((j, row[j]))

        for i in range(len(ady_list)):
            for j in range(len(ady_list[i])):
                y = ady_list[i][j][0]
                weight = ady_list[i][j][1]
                G.add_edge(self.list_nodes[y], self.list_nodes[i], weight=weight)

        self.G_small = G

        nx.write_edgelist(G, f"0set_size{str(size)}.csv", delimiter=",", data=['weight'])
        nx.write_gexf(G, f"0set_size{str(size)}.gexf")
    
    def export_graph_to_graphml(self, path = "./dataset/graph_19k_3.5m.gml"):
        nx.write_graphml(self.G, path)

    def export_graph_to_adjlist(self, path = "./dataset/graph_19k_3.5m.adyl"):
        nx.write_adjlist(self.G, path)

    def load_ady_matrix(self, count = 0):

        for x in range(count+1):
            print(f'Loading: {x} file.csv')
            df = pd.read_csv(f'./dataset/matrix/{x} file.csv')            
            for i in range(df.shape[0]):
                row = df.loc[i].to_list()
                for j in range(len(row)):
                    if row[j] != 0:
                        self.ady_list[i + x * 642].append((j, row[j]))
            print(f'Finished: {x} file.csv')


        with open(f'./dataset/adym_{count}.pkl', 'wb') as f:
            pickle.dump(self.ady_list, f)
   
    def save_graph_obj(self, path = './dataset/graph-1.x.pkl'):
        
        with open(path, 'wb') as f:
            pickle.dump(self.G, f)

    def load_matrix_obj(self, path = './dataset/graph_19k_3.5m.pkl'):
        self.G = pickle.load(open(path, 'rb'))
    
    def read_adym(self, path = './dataset/adym_30.pkl'):
        self.ady_list = pickle.load(open(path, 'rb'))

    def save_attributed_graph(self, path = './dataset/outputs/attributed_graph.csv'):
        '''
        Save the graph with the attributes of each node in csv file.
        '''

        with open(path, 'w') as f:
            for node in self.G.nodes:
                
                f.write(f'{node},{str(nx.degree(self.G, node))},{str(self.G.in_degree[node])},{str(self.G.out_degree[node])},')
                f.write(str(self.G.degree(node, weight='weight')) + ',')
                f.write(str(self.G.in_degree(node, weight='weigth')) + ',')
                f.write(str(self.G.out_degree(node, weight='weigth')) + ',')
                f.write(str(self.G.nodes[node]['eigenvector_centrality']) + ',')
                f.write(str(self.G.nodes[node]['eigenvector_centrality_weighted']) + ',')
                f.write(str(self.G.nodes[node]['pagerank']) + ',')
                f.write(str(self.G.nodes[node]['degree_centrality']) + ',')
                f.write(str(self.G.nodes[node]['core_number']) + '\n')


    # ALGORITMOS DE COMUNIDADES

    def lovain_concurrent(self, weight = 'weight', resolution = 1, threshold = 1e-07, seed = None , n = 10):

        '''
        This functiosn is for execute louvain algorithm in parallel.

        Parameters
        ----------
        G : NetworkX graph
        weight : string or None, optional (default="weight")
            The name of an edge attribute that holds the numerical value
            used as a weight. If None then each edge has weight 1.
        resolution : float, optional (default=1)
            If resolution is less than 1, the algorithm favors larger communities.
            Greater than 1 favors smaller communities
        threshold : float, optional (default=0.0000001)
            Modularity gain threshold for each level. If the gain of modularity
            between 2 levels of the algorithm is less than the given threshold
            then the algorithm stops and returns the resulting communities.
        seed : list (lenght=n), random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
        n :int, optional (default=10)
            Number of times to execute the algorithm.

        Returns
        -------
        list 
            A list of sets (partition of `G`). Each set represents one community and contains
            all the nodes that constitute it.
        '''

        import networkx.algorithms.community as nx_comm

        if seed:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(nx_comm.louvain_communities, [(self.G, weight, resolution, threshold, seed[i]) for i in range(n)])
        else:            
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(nx_comm.louvain_communities, [(self.G, weight, resolution, threshold) for _ in range(n)])

        return communities

    def lpa_wrapper(self, G, weight = 'weight', seed = 1):

        import networkx.algorithms.community as nx_comm
        return list(nx_comm.asyn_lpa_communities(G, weight, seed))

    def asyn_lpa_concurrent(self, weight = 'weight', seed = None , n = 10):
        
        '''  
        This functiosn is for execute asyn_lpa algorithm in parallel.

        Parameters
        ----------
        G : Graph

        weight : string
            The edge attribute representing the weight of an edge.
            If None, each edge is assumed to have weight one. In this
            algorithm, the weight of an edge is used in determining the
            frequency with which a label appears among the neighbors of a
            node: a higher weight means the label appears more often.

        seed : list(integer) with length = n , random_state, or None (default = 1)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
        
        n :int, optional (default=10)
            Number of times to execute the algorithm.

        Returns
        -------
        communities : iterable
            Iterable of communities given as sets of nodes.

        Notes
        -----
        Edge weight attributes must be numerical.

        '''

        if seed:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(self.lpa_wrapper, [(self.G, weight, seed[i]) for i in range(n)])
        else:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(self.lpa_wrapper, [(self.G, weight) for _ in range(n)])
        

        return [com for com in communities]

    def greedy_modularity_concurrent(self, weight=None, resolution=1, cutoff=1, best_n=None, n = 10):
        
        '''
        This functiosn is for execute greedy modularity algorithm in parallel.

        Parameters
        ----------
        G : NetworkX graph

        weight : string or None, optional (default=None)
            The name of an edge attribute that holds the numerical value used
            as a weight.  If None, then each edge has weight 1.
            The degree is the sum of the edge weights adjacent to the node.

        resolution : float, optional (default=1)
            If resolution is less than 1, modularity favors larger communities.
            Greater than 1 favors smaller communities.

        cutoff : int, optional (default=1)
            A minimum number of communities below which the merging process stops.
            The process stops at this number of communities even if modularity
            is not maximized. The goal is to let the user stop the process early.
            The process stops before the cutoff if it finds a maximum of modularity.

        best_n : int or None, optional (default=None)
            A maximum number of communities above which the merging process will
            not stop. This forces community merging to continue after modularity
            starts to decrease until `best_n` communities remain.
            If ``None``, don't force it to continue beyond a maximum.

        n :int, optional (default=10) 
            Number of times to execute the algorithm. 
        
        Returns:
            list (frozenset): A list of sets (partition of G). Each set represents one community and contains all the nodes that constitute it.
        
        '''
        
        
        import networkx.algorithms.community as nx_comm

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            communities = pool.starmap(nx_comm.greedy_modularity_communities, [(self.G, weight ,resolution, cutoff,best_n) for _ in range(n)])
        return communities
    
    def infomap_concurrent(self, seed = None, n = 10):

        number_of_pool = n if n < multiprocessing.cpu_count() else multiprocessing.cpu_count()
        '''
        This functiosn is for execute infomap algorithm in parallel.
        Parameters
        ----------

            G (networkx.Graph): Graph to be clustered.

            n (int, optional): Number of times to execute the algorithm. Defaults to 10.

            seed : list(integer) with length = n , random_state, or None (default = 1)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns:
        ----
            list (cdlib.classes.node_clustering.NodeClustering): List of communities.
                
        NodeClustering type Properties:
            communities: List of communities
            graph: A networkx/igraph object
            method_name: Community discovery algorithm name
            method_parameters: Configuration for the community discovery algorithm used
            overlap: Boolean, whether the partition is overlapping or not
        '''
        from cdlib import algorithms

        if seed:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(algorithms.infomap, [(self.G, '--seed ' + f'{seed[i]}') for i in range(n)])
        else:
            with multiprocessing.Pool(number_of_pool) as pool:
                communities = pool.map(algorithms.infomap, [self.G for _ in range(n)])

        return communities
    # Tools for Algorithms

    def communities_length(self, communities):

        '''
        This function is for calculate the length of each partition in the community.
        '''

        a = [len(com) for com in communities]
        a.sort(reverse=True)

        return a
    
    def save_communities(self, communities, algorithm : str, params: list = [], seed = None):

        if algorithm == 'louvain':
            
            if params == []:
                params = [1, 1e-07, None]

            for i in range(len(communities)):
                params_name = ''.join(['_' + str(param) for param in params])
                params_name += '_seed_' + str(seed[i]) if seed else ''

                with open('./dataset/outputs/' + algorithm + '/' + algorithm + params_name + '_iter_' + str(i) , 'wb+') as f:
                    pickle.dump(communities[i], f)

        elif algorithm == 'greedy':

            if params == []:
                params = [1, 1, 0]

            for i in range(len(communities)):
                params_name = ''.join(['_' + str(param) for param in params])

                with open('./dataset/outputs/' + algorithm + '/' + algorithm + params_name + '_iter_' + str(i) , 'wb+') as f:
                    pickle.dump(communities[i], f)

        elif algorithm == 'lpa':

            for i in range(len(communities)):

                params_name = ''
                params_name += '_seed_' + str(seed[i]) if seed else ''

                with open('./dataset/outputs/' + algorithm + '/' + algorithm + params_name + '_iter_' + str(i) , 'wb+') as f:
                    pickle.dump(communities[i], f)

        elif algorithm == 'infomap':

            for i in range(len(communities)):

                params_name = ''
                params_name += '_seed_' + str(seed[i]) if seed else ''

                with open('./dataset/outputs/' + algorithm + '/' + algorithm + params_name + '_iter_' + str(i) , 'wb+') as f:
                    pickle.dump(communities[i], f)


    def load_communities(self, algorithm : str, resolution = 1, threshold = 1e-07 , seed = 1, iter = 0) -> list:

        if algorithm == 'louvain':
            path = './dataset/outputs/' + algorithm + '/' + algorithm + '_'+ str(resolution) + '_' + str(threshold) + '_seed_' + str(seed) + '_iter_' + str(iter)
        
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        return []

    def load_all_communities(self, algorithm : str) -> list: 
        '''
        This function is for load all the communities generated by one algorithm.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to load the communities.
        
        Returns
        -------
        list : list
            A list of communities present in the folder of the algorithm.
        '''

        if algorithm == 'louvain':
            
            paths = os.listdir('./dataset/outputs/' + algorithm + '/')
        
            communities = []
            for path in paths:
                with open('./dataset/outputs/' + algorithm + '/' + path, 'rb') as f:
                    community = pickle.load(f)
                    sorted_community = sorted(community, key = lambda x: len(x), reverse = True)
                    communities.append(sorted_community)

            return communities
        if algorithm == 'lpa':

            paths = os.listdir('./dataset/outputs/' + algorithm + '/')
        
            communities = []
            
            for path in paths:
                with open('./dataset/outputs/' + algorithm + '/' + path, 'rb') as f:
                    communities.append(pickle.load(f))
            
            return communities
        
        if algorithm == 'greedy':

            paths = os.listdir('./dataset/outputs/' + algorithm + '/')

            communities = []

            for path in paths:
                with open('./dataset/outputs/' + algorithm + '/' + path, 'rb') as f:
                    communities.append(pickle.load(f))
        
            
            return communities

        if algorithm == 'infomap':

            paths = os.listdir('./dataset/outputs/' + algorithm + '/')

            communities = []

            for path in paths:
                with open('./dataset/outputs/' + algorithm + '/' + path, 'rb') as f:
                    communities.append(pickle.load(f))
        
            
            return communities
        
        return []    

    def compare_communities_limits(self, communities, limit: list = [1,1]):

        '''
        This function is for compare the communities generated by an algorithm with a limit.

        Parameters
        ----------
        communities : list
            A list of communities generated by one algorithm.
        limit : list (default = [1,1])
            A list of two elements. The first element is the cuantity of the biggest communities (by length) must be compared.
            The second element is the cuantity of the smallest communities (by length) must be compared.
        
        Returns
        -------
        list : list with two elements. 
            A list of elemments .
        '''

        return None
    
    def create_sub_graph(self, community):

        '''
        This function is for create a subgraph from one community.

        Parameters
        ----------
        community : list
            A list of nodes that form a community.
        
        Returns
        -------
        Graph : NetworkX DiGraph
            A subgraph of the original graph.
        '''

        return self.G.subgraph(community)
    
    def add_degree_property(self, G):

        '''
        This function is for add a property to the nodes of the graph. The property is the degree of the node.

        Parameters
        ----------
        G : NetworkX DiGraph
            A graph.
        
        Returns
        -------
        Graph : NetworkX DiGraph
            A graph with the property degree.
        '''

        for node in G.nodes():
            G.nodes[node]['degree'] = G.degree(node)

        return G

    def nodes_in_communities(self, communities: list):

        '''
        This function is for return a dict with the name of the nodes is teh key and the values
         is the communities to they belong.

        Parameters
        ----------
        communities : list
            A list of communities generated by one algorithm.
        
        Returns
        -------
        result : dict
            A dict of nodes in their communities.
        '''

        nodes = {}
        for i in range(len(communities)):
            for node in communities[i]:
                nodes[node] = i

        return nodes

    def save_dict_to_csv(self, dict, name):

        '''
        This function is for save a dict to a csv file.

        Parameters
        ----------
        dict : dict
            A dict to save.
        name : str
            The name of the csv file.
        '''

        with open('./dataset/outputs/graph/' + name + '.csv', 'w') as f:
            for key, value in dict.items():
                f.write(str(key) + ',' + str(value) + '\n')

    def add_property(self, measure: list):

        '''
        This function is for add a property to the nodes of the graph.

        Parameters
        ----------
        property : str
            The name of the property to add.
        '''

        if 'eigenvector_centrality' in measure:

            data = nx.eigenvector_centrality(self.G)

            data_weighted = nx.eigenvector_centrality(self.G, weight='weight') 

            for node in self.G.nodes():
                self.G.nodes[node]['eigenvector_centrality'] = data[node] # type: ignore
                self.G.nodes[node]['eigenvector_centrality' + '_weighted'] = data_weighted[node] # type: ignore
            
            print('eigenvector_centrality added')
        
        if 'pagerank' in measure:

            data = nx.pagerank(self.G)           

            for node in self.G.nodes():
                self.G.nodes[node]['pagerank'] = data[node]

            print('pagerank added')
        
        if 'degree_centrality' in measure:

            data = nx.degree_centrality(self.G)           

            for node in self.G.nodes():
                self.G.nodes[node]['degree_centrality'] = data[node]

            print('degree_centrality added')
        
        if 'core_number' in measure:

            data = nx.core_number(self.G)           

            for node in self.G.nodes():
                self.G.nodes[node]['core_number'] = data[node] # type: ignore
            
            print('core_number added')
        
    def participation_coefficient(self, communities: list):

        '''
        This function is for calculate the participation coefficient of a community.

        Parameters
        ----------        
        communities : list
            A list of comunities.
        
        Returns
        -------
        result : dict
            The participation coefficient of every node.
        '''

        data_nodes = {}
        
        start_time = datetime.datetime.now()

        count = 0
        
        for node in self.G.nodes():

            k_i = self.G.degree(nbunch=node)

            suma = 0
            
            neighbors = list(nx.neighbors(self.G, node))

            for community in communities:

                subgraph = nx.subgraph(self.G, community)
                                
                k_i_s = 0

                for n in neighbors:
                    
                    count += 1
                    if count == 10000000:
                        print('10M operations calculated in participation coefficient')
                        print('Time for 10M operations: ' + str(datetime.datetime.now() - start_time))
                        count = 0
                        start_time = datetime.datetime.now()

                    if n in subgraph.nodes():
                        k_i_s += 1
                        continue

                suma += (k_i_s / k_i) ** 2 # type: ignore

            data_nodes[node] = 1 - suma

        return data_nodes

    def insert_measure_dict(self, measure: str, dict: dict):

        '''
        This function is for insert a measure to the nodes of the graph.

        Parameters
        ----------
        measure : str
            The name of the measure.
        dict : dict
            A dict with the values of the measure.
        '''

        for node in self.G.nodes():
            self.G.nodes[node][measure] = dict[node]
    
    # Begining of Horacio's Region

    def edgeWithinComm(self, vi, si, w, directed):
        sugSi = nx.subgraph(self.G, si)
        if not directed:
            ki = sugSi.degree(vi, weight = w)
        else:
            ki = sugSi.out_degree(vi, weight = w)
        return ki

    def degMeanByCommunity(self, si, w):
        sum = 0
        for vi in si:
            edgeWith = self.edgeWithinComm(vi, si, w)
            sum += edgeWith
        return sum/len(si)

    def standDesvCommunity(self, si, w):
        dMeanCi = self.degMeanByCommunity(si, w)
        sum = 0 
        for vi in si:
            ki = self.edgeWithinComm(vi, si, w)
            sum += math.pow((ki - dMeanCi), 2)
        desv = math.sqrt(sum/len(si))
        return desv

    def withinCommunityDegree(self, w, commList, algName, directed):
        zi = dict()
        run_community_mean = dict()
        run_community_vi = dict()
        run_community_desv = dict()

        # iterate through all runs
        for run_i in range(0, len(commList)):
            clustering_i = commList[run_i]
            # iterate through all commnunities
            for c_j in range(0, len(clustering_i)):
                sumMean = 0
                key_RunComm = (run_i, c_j)
                for vi in clustering_i[c_j]:
                    key_RunCommVi = (run_i, c_j, vi)
                    if key_RunCommVi not in run_community_vi:
                        ki = self.edgeWithinComm(vi, clustering_i[c_j], w, directed)
                        run_community_vi[key_RunCommVi] = ki
                        sumMean += ki
                if key_RunComm not in run_community_mean:
                    ciMean = sumMean/len(clustering_i[c_j])
                    run_community_mean[key_RunComm] = ciMean
        # ki, mean(si) -> DONE
        print('ki, mean(si) -> DONE')

        # iterate through all runs
        for run_i in range(0, len(commList)):
            clustering_i = commList[run_i]
            # iterate through all commnunities
            for c_j in range(0, len(clustering_i)):
                sumDesv = 0
                key_runC = (run_i, c_j)
                meanCj = run_community_mean[key_runC]
                for vi in clustering_i[c_j]:
                    key_RunCVi = (run_i, c_j, vi)
                    kVi = run_community_vi[key_RunCVi]
                    sumDesv += math.pow((kVi - meanCj), 2)
                if key_runC not in run_community_desv:
                    desvCj = math.sqrt(sumDesv/len(clustering_i[c_j]))
                    run_community_desv[key_runC] = desvCj
        print('desviation -> DONE')
        # print(run_community_vi)

        
        # iterate through all vertex
        for vi in self.G.nodes:
            # iterate through all runs
            for run_i in range(0, len(commList)):
                clustering_i = commList[run_i]
                # iterate through all commnunities
                for c_j in range(0, len(clustering_i)):
                    # identify the Ci 
                    if vi in clustering_i[c_j]:
                        desvSi = run_community_desv[(run_i, c_j)]
                        ziValue = 0
                        if desvSi != 0:
                            meanSi = run_community_mean[(run_i, c_j)]
                            ki = run_community_vi[(run_i, c_j, vi)]
                            ziValue = (ki - meanSi)/desvSi
                        if vi not in zi:
                            listTupleValue = []
                            listTupleValue.append(ziValue)
                            zi[vi] = listTupleValue
                        else:
                            zi[vi].append(ziValue)
                        break
        
            # count+=1
            # print('vertex: ', vi, ' Done', ' numbers of items: ', count)
        print('zi -> DONE')

        direct = 'directed_' if directed else 'notDirected_'
        weg = 'weighted' if w == 'weight' else 'notWeighted'
        nameFile = 'within_' + algName + '_' + direct + weg
        # create a binary pickle file 
        f = open('./dataset/outputs/' + nameFile ,"wb")

        # write the python object (dict) to pickle file
        pickle.dump(zi, f)

        # close file
        f.close()

    def calculateWithin(self, algList = ['louvain', 'greedy', 'lpa', 'infomap'], wegList = ['weight', 'none'], directList = [True, False]):
        for alg_i in algList:
            for weg_j in wegList:
                for direct_k in directList:
                    commList =  m.load_all_communities(alg_i)
                    if alg_i == 'infomap':
                        commListModified = []
                        for ci in commList:
                            commListModified.append(ci['communities'])
                        self.withinCommunityDegree(weg_j, commListModified, alg_i, direct_k)
                    else:
                        self.withinCommunityDegree(weg_j, commList, alg_i, direct_k)
                    print('finish Directed: ', direct_k)
                print('finish Weight: ', weg_j)
            print('finish Algorithm: ', alg_i)

    def edgeMetricsExport(self, pathFile):
        with open(pathFile, 'w') as f:
            for u,v in self.G.edges:
                f.write(u + ', ' + v + ', ' + str(self.G.edges[u,v]['weight']) + ', ' + str(self.G.edges[u,v]['edge_betweenes']))
                f.write('\n')
                
        

        




    # End Horacio's Region

    def apply_measures_to_communities_nodes(self, algorithm: str ,communities: list):
        
        

        for i in range(len(communities)):            
                
            start_time = datetime.datetime.now()
            iter_number = i
            iter_communities = communities[i]['communities'] if algorithm == 'infomap' else communities[i]
            data_participation = {}
            community_number = 0

            data_participation = self.participation_coefficient(iter_communities)

            print('Participation coefficient calculated for community ' + str(i) + ' of ' + str(len(communities)) + ' run of ' + algorithm + ' algorithm')
            print('Time for participation coefficient: ' + str(datetime.datetime.now() - start_time))

            for k, v in data_participation.items():

                for j in range(len(iter_communities)):
                    if k in iter_communities[j]:
                        community_number = j
                        break

                if 'data' not in self.G.nodes[k].keys():
                    self.G.nodes[k]['data'] = {(algorithm, str(iter_number), str(community_number)): {'participation_coefficient': v}} # type: ignore
                else:
                    if (algorithm, str(iter_number), str(community_number)) not in self.G.nodes[k]['data'].keys():
                        self.G.nodes[k]['data'][(algorithm, str(iter_number), str(community_number))] = {'participation_coefficient': v} # type: ignore
                    else:
                        self.G.nodes[k]['data'][(algorithm, str(iter_number), str(community_number))]['participation_coefficient'] = v
        
        # Withing Region

        # start_time = datetime.datetime.now()       
        # data_whiting_degree = self.withinCommunityDegree('None', communities)
        # print('Time for whiting degree: ' + str(datetime.datetime.now() - start_time))
        
        # start_time = datetime.datetime.now()
        # data_withing_degree_weighted = self.withinCommunityDegree('weight', communities)
        # print('Time for whiting degree weighted: ' + str(datetime.datetime.now() - start_time))

        # self.save_attributed_graph()
        # print('Graph saved with whiting degree and participation coefficient')
        # print('*******************************************************')
        
        # community_number = 0

        # for k, v in data_whiting_degree.items():

        #     for j in range(len(v)):
        #         for i in range(len(communities[j])):
        #             if k in communities[j][i]:
        #                 community_number = i
        #                 break
                                
        #         if 'data' not in self.G.nodes[k].keys():
        #             self.G.nodes[k]['data'] = {(algorithm, str(j), str(community_number)): {'whiting_degree': v[j]}} # type: ignore        
        #         else:
        #             if (algorithm, str(j), str(community_number)) not in self.G.nodes[k]['data'].keys():
        #                 self.G.nodes[k]['data'][(algorithm, str(j), str(community_number))] = {'whiting_degree': v[j]}
        #             else:
        #                 self.G.nodes[k]['data'][(algorithm, str(j), str(community_number))]['whiting_degree'] = v[j]

        # for k, v in data_withing_degree_weighted.items():

        #     for j in range(len(v)):
        #         for i in range(len(communities[j])):
        #             if k in communities[j][i]:
        #                 community_number = i
        #                 break
        #         if 'data' not in self.G.nodes[k].keys():
        #             self.G.nodes[k]['data'] = {(algorithm, str(j), str(community_number)): {'whiting_degree_weighted': v[j]}} # type: ignore        
        #         else:
        #             if (algorithm, str(j), str(community_number)) not in self.G.nodes[k]['data'].keys():
        #                 self.G.nodes[k]['data'][(algorithm, str(j), str(community_number))] = {'whiting_degree_weighted': v[j]}
        #             else:
        #                 self.G.nodes[k]['data'][(algorithm, str(j), str(community_number))]['whiting_degree_weighted'] = v[j]

        # end Whiting Region
                    
                        

    def apply_measures_to_communities(self, communities: list):                   
        pass
        # conductance_dict = {}

        #     for ci in range(len(iter_communities)):

        #         for cj in range(ci + 1, len(iter_communities)):

        #             community_i = iter_communities[ci]
        #             community_j = iter_communities[cj]

        #             data_conductance = nx.conductance(self.G, community_i, community_j, weight='weight')
                
        #             conductance_dict[(ci, cj)] = data_conductance
        


def save_all_communities_tocsv(algorithm: str, communities: list):

    '''
    This function is for save all the communities generated by one algorithm to a csv file.

    Parameters
    ----------
    algorithm : str
        The name of the algorithm that was used to save the communities.
    '''

    df = pd.read_csv('dataset/outputs/all.csv')
    
    df.set_index('id', inplace=True)

    if algorithm != 'infomap':
    

        for i in range(len(communities)):

            for z in range(len(communities[i])):
                
                for _,value in enumerate(communities[i][z]):
                    df.loc[value, i] = z

        for i in range(len(communities)):
            df[i] = df[i].astype('Int64')
    else:

        for i in range(len(communities)):

            for z in range(len(communities[i].communities)):
                for _,value in enumerate(communities[i].communities[z]):
                    df.loc[value, i] = z

        for i in range(len(communities)):
            df[i] = df[i].astype('Int64')

    df.to_csv('dataset/outputs/all_' + algorithm + '.csv')
    
                
def writter(lis, name):

    with open('./dataset/outputs/' + name, 'w') as f:
        for (id, value) in lis:
            f.write(f'{id}, {value}\n')  

def run_and_save_algorithm(m: Matrix, algorithm, params, n, seed = []) :

    if seed == []:
        seed = [x for x in range(n)]

    if algorithm == 'louvain':

        communities = m.lovain_concurrent(seed= seed,  n=10)

        for com in communities:
            print(m.communities_length(com))

        m.save_communities(communities, 'louvain', params=params, seed= seed)
    
    elif algorithm == 'greedy':

        communities = m.greedy_modularity_concurrent(resolution=2 , n=n)

        for com in communities:
            print(m.communities_length(com))

        m.save_communities(communities, 'greedy', params=params )
    
    elif algorithm == 'lpa':

        communities = m.asyn_lpa_concurrent(weight = 'weight', seed = seed , n = n)

        for com in communities:
            print(m.communities_length(com))

        m.save_communities(communities, 'lpa', params=params, seed = seed )

    elif algorithm == 'infomap':

        communities = m.infomap_concurrent(seed=seed, n = n)

        for com in communities:
            print(m.communities_length(com.communities))

        m.save_communities(communities, 'infomap', params=params, seed = seed )
      
def small(communities: list):
    
    com = []
    data = {}

    for comm, path in communities:

        data['communities'] = comm.communities
        data['method_name'] = comm.method_name
        data['method_parameters'] = comm.method_parameters
        data['overlap'] = comm.overlap

        com.append((data, path))
        data = {}

    return com

   
    


if __name__ == '__main__':
    

    measures = ['eigenvector_centrality', 'pagerank', 'degree_centrality', 'core_number']

    m = Matrix([], {},[])
    #m.load_matrix_obj()
    #m.export_graph_to_adjlist()
    #m.insert_nodes()
    #m.read_adym(path='data/adym_0.pkl')
    #m.read_adym(path='dataset/adym_30.pkl')
    #m.load_ady_matrix(30)    
    #m.insert_weighted_edges()
    # m.sava_matrix_obj()
    m.load_matrix_obj(path='dataset/attributed_graph-1.4.fly')

    print(m.G.number_of_edges())

    print(datetime.datetime.now())

    m.edgeMetricsExport(pathFile='output/edgesMEasure.csv')

    # run_and_save_algorithm(m, 'infomap', params= [], seed=[1,2,3,4,5,6,7,8,9,10], n= 10)
    
    # communities = algorithms.infomap(m.G, flags='--seed 23').communities

    # sorted_community = sorted(communities, key=lambda x: len(x), reverse=True)

    # print(m.communities_length(sorted_community))

        
    # community = communities[0]

    # sorted_community = sorted(community, key=lambda x: len(x), reverse=True)

    # for i in range(len(community)):        
    #         print(len(sorted_community[i]))
    

    # m.apply_measures_to_communities_nodes('greedy', communities)

    # communities = m.load_all_communities('lpa')

    # m.apply_measures_to_communities_nodes('lpa', communities)

    # communities = m.load_all_communities('infomap')

    # m.apply_measures_to_communities_nodes('infomap', communities)

    # m.save_graph_obj(path='dataset/attributed_graph-1.3.pkl')

    # print(m.G.nodes['104198-F-000008'])

    print(datetime.datetime.now())
    
    

    # print(sorted(data.items(), key=lambda x: x[1], reverse=True))

    # print(datetime.datetime.now())   
    
    #save_all_communities_tocsv('lpa', communities)

    #communities = small(communities)

    # for com,path in communities:
    #      with open('./dataset/outputs/lpa' + path , 'wb+') as f:
    #                 pickle.dump(com, f)

    # for i  in range(len(communities)):        

    #     community = communities[i]

    #     sorted_community = sorted(community, key=lambda x: len(x), reverse=True)

    #     print(m.communities_length(sorted_community))
        

        #data = m.nodes_in_communities(sorted_community)

        #m.save_dict_to_csv(data, 'louvain_1_1e-07_seed_1_iter_' + str(i))

    # community = m.load_communities('louvain')

    # sorted_community = sorted(community, key=lambda x: len(x), reverse=True)

    # SG = m.create_sub_graph(sorted_community[0])

    # nx.write_edgelist(SG, './dataset/outputs/graph/louvain_1_1e-07_seed_1_iter_0[2897].csv', delimiter=",", data=['weight'])

    #SG = m.add_degree_property(SG)

    #nx.write_graphml(SG, './dataset/outputs/louvain/louvain_1_1e-07_seed_1_iter_0[2897].graphml')
    
        
    # coun_int = 0
    # coun_out = 0    
    
    
    # sorted_degree = sorted(list(m.G.degree()), key=lambda x: x[1], reverse=True)

    # for i in range(0, 9700):
    #     node = sorted_degree[i][0]
    #     if m.G.in_degree(node) >  m.G.out_degree(node):
    #         coun_int += 1
    #     else:
    #         coun_out += 1

    # print(coun_int, coun_out)


    # sorted_weighted = sorted(list(m.G.degree(weight='weight')), key=lambda x: x[1], reverse=True)

    # set_degree = set()
    
    # for i in range(0, 9700):
    #     set_degree.add(sorted_degree[i][0])

    # coun = 0

    # for i in range(0, 9700):
    #     if sorted_weighted[i][0] in set_degree:
    #         coun += 1

    # print(coun)
    
    #print(sum([x[1] for x in sorted_degree[0:3995]]))

    #de_cen = nx.degree_centrality(m.G)

    #print(de_cen)
    #print(sorted(de_cen.items(), key=lambda x: x[1]))

    #print(m.G.degree('TH-F-300002'))

    # a = m.G.degree(weight='weight')
    
    # writter(a, 'weight_degree_30.csv')
    
    # a = m.G.in_degree(weight='weight')
    
    # writter(a, 'weight_in_degree_30.csv')    

    # a = m.G.out_degree(weight='weight')
    
    # writter(a, 'weight_out_degree_30.csv')
    
    #print(sorted(list(a), key=lambda x: x[1], reverse=True))

    #print(nx.betweenness_centrality(m.G))

    #print(sorted(a., key=lambda x: x[1]))
    #hist = nx.degree_histogram(m.G)
    #print(hist)

    # co = 0
    # for i in range(len(hist)):
    #     if hist[i] != 0:
    #         co += hist[i]*i
    # print(co)

    # list_hist = []

    # for i in range(len(hist)):
    #     if hist[i] != 0:
    #         list_hist.append(i)

    # plt.hist(list_hist)
    # plt.savefig('hist.png')

    # var = []

    # for i in range(len(list_hist)):
    #     if list_hist[i] >= 1500:
    #         var.append(i)

    # list_hist = []

    # for i in range(len(hist)):
    #     if hist[i] != 0:
    #         if hist[i] == 1:
    #             list_hist.append(i)
    #         else:
    #             list_hist.append(i * hist[i] / 2)
    #     else:
    #         list_hist.append(0)
    
    # print(sum(list_hist))
    # print(len(var))
    # print(sum(var) / 2)
    #print(nx.function.degree_histogram(m.G))

    #ouliers = [['E0585-F-300022'], ['Gad1-F-300464'], ['VGlut-F-100031'], ['VGlut-F-100283'], ['VGlut-F-400694']]

    # for node in ouliers:
    #     id = m.pos_code_nodes[node[0]]
    #     row = m.ady_list[id]
    #     value = 0
    #     suma = 0
    #     for x in row:
    #         value += x[1]
    #     print(f'Node: {node} sum values: {value} : len : {len(m.ady_list[id])}' )
    #     print(len(m.G.adj[node[0]].items()))
    a = []
    
    time1 = time.time()

    #comm = nx_comm.asyn_lpa_communities(m.G, weight='weight', seed=5)
     

    # for x in comm:       
    #     a.append(len(x))
    #     a.sort()
    # print(a)
    # time2 = time.time()
    # print(time2 - time1)
    
    # comm = nx_comm.greedy_modularity_communities(m.G, weight='weight')

    # time2 = time.time()
    # print(time2 - time1)

    # for x in comm:       
    #     a.append(len(x))
    #     a.sort()
    # print(a)

    # comm = nx_comm.greedy_modularity_communities(m.G, weight=None)

    # time2 = time.time()
    # print(time2 - time1)

    # for x in comm:       
    #     a.append(len(x))
    #     a.sort()
    # print(a)

    #m.export_graph_to_csv()
    # m.load_ady_matrix(30)
    # for node in m.list_nodes:
    #     count += 1
    #     if count < 10:
    #         print(f'{node} : {m.G.degree[node]}')
    #         print(f'{node} : {m.G.adj[node]}')
    #     else:
    #         break
    #nx.write_gexf(m.G, "0set.gexf")
    
    #m.export_graph_to_csv_size(500)
    #print(m.G.number_of_edges())
    # print(len(list(m.G.neighbors('104198-F-000000'))))

    # print(len(list(nx.strongly_connected_components(m.G))))      
   
    # print(len(q.get().communities))
    # print(q.get().method_name)

    # print(len(q.get().communities))
    # print(q.get().method_name)

    # for pro in procs:
    #     pro.join()

    # time2 = time.time()
    # print(time2 - time1)
    # singles_nodes = []

    
    # for q in coms.communities:       
    #     a.append(len(q))
    #     a.sort()
    #     if len(q) == 1:
    #         singles_nodes.append(q)
    # print(a)
    # a = []
    # print(singles_nodes)
    # singles_nodes = []
    
    
    
    # start = time.time()
    # b = nx_comm.louvain_communities(m.G, seed= 1)
    # end = time.time()
    # print(end - start)

    # for g in b:       
    #     a.append(len(g))
    #     a.sort()
    
    # print(a)