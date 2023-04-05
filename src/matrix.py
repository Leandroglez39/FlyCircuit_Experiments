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
import numpy as np
import scipy.stats as stats
import statistics





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
        nx.write_edgelist(self.G, path, delimiter=",", data=['weight']) # type: ignore

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

        nx.readwrite.edgelist.write_edgelist(G, f"0set_size{str(size)}.csv", delimiter=",", data=['weight'])        
        nx.gexf.write_gexf(G, f"0set_size{str(size)}.gexf")
        
    def export_graph_to_graphml(self, path = "./dataset/graph_19k_3.5m.gml"):        
        nx.graphml.write_graphml(self.G, path)

    def export_graph_to_adjlist(self, path = "./dataset/graph_19k_3.5m.adyl"):        
        nx.adjlist.write_adjlist(self.G, path)

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
                
                f.write(f'{node},{str(nx.degree(self.G, node))},{str(self.G.in_degree[node])},{str(self.G.out_degree[node])},') # type: ignore
                f.write(str(self.G.degree(node, weight='weight')) + ',') # type: ignore
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
        return list(nx_comm.asyn_lpa_communities(G, weight, seed)) # type: ignore
        

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
        from cdlib import algorithms # type: ignore Only necesary with cdlib environment #TODO

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
            
            paths = os.listdir('./output/' + algorithm + '/')
        
            communities = []
            for path in paths:
                with open('./output/' + algorithm + '/' + path, 'rb') as f:
                    community = pickle.load(f)
                    sorted_community = sorted(community, key = lambda x: len(x), reverse = True)
                    communities.append(sorted_community)

            return communities
        if algorithm == 'lpa':

            paths = os.listdir('./output/' + algorithm + '/')
        
            communities = []
            
            for path in paths:
                with open('./output/' + algorithm + '/' + path, 'rb') as f:
                    communities.append(pickle.load(f))
            
            return communities
        
        if algorithm == 'greedy':

            paths = os.listdir('./output/' + algorithm + '/')

            communities = []

            for path in paths:
                with open('./output/' + algorithm + '/' + path, 'rb') as f:
                    communities.append(pickle.load(f))
        
            
            return communities

        if algorithm == 'infomap':

            paths = os.listdir('./output/' + algorithm + '/')

            communities = []

            for path in paths:
                with open('./output/' + algorithm + '/' + path, 'rb') as f:
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

            data = nx.core_number(self.G)            # type: ignore

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

            k_i = self.G.degree(nbunch=node) # type: ignore

            suma = 0
            
            neighbors = list(nx.neighbors(self.G, node)) # type: ignore

            for community in communities:

                subgraph = nx.subgraph(self.G, community) # type: ignore
                                
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
    

    def RoughClustering(self, communities: list):

        '''
        This function is for calculate the rough clustering of a community list.

        Parameters
        ----------        
        communities : list
            A list of list of comunities.
        
        Returns
        -------
        result : dict
            The rough clustering of every node.
        '''
        

         # Create a list of nodes from the graph
        nodes = list(self.G.nodes())
        
        # Initialize a dict to store the nodes in the community
        dict_nodes = {}
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                keyi = (nodes[i],nodes[j])           
                dict_nodes[keyi] = 0
        
                
        Gb0 = nx.Graph()

        Gb0.add_nodes_from(nodes)
              

        match_dict = {}
        
        for community in communities:
            temp_match = self.update_dict_count(dict_node=dict_nodes, community=community)
            self.merge_communities_dict(match_dict, temp_match)

        b0 = len(communities)/2 - 1

        edges_list = []

        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                keyi = (nodes[i],nodes[j])           
                if match_dict[keyi] >= b0:
                    edges_list.append(keyi)
        
        
        
        Gb0.add_edges_from(edges_list)

        strongcomponents = nx.connected_components(Gb0)

        seeds = []

        for component in strongcomponents:
            seeds.append(nx.function.subgraph(self.G, component))
            

        seeds.sort(key=len, reverse=True)

        k = self.calculate_k(communities)

        print

        return seeds

    
    def update_dict_count(self, dict_node: dict ,community: list):
        

        '''
        This function is for update the dict of the count of the communities.

        Parameters
        ----------
        community : list
            A list of nodes of a community.
        '''
        
        dict_nodes = dict_node.copy()

        # For each community
        for com in community:
            # For each node in the community
            com.sort()
            for i in range(len(com)):
                nodei = com[i]
                # For each node in the community except the current node
                for j in range(i+1, len(com)):
                    nodej = com[j]
                    # If the current combination of nodes is not in the dict
                    if (nodei, nodej) not in dict_nodes:
                        # Add it with a count of 1
                        dict_nodes[(nodei,nodej)] = 1
                    else:
                        # If it is in the dict, add 1 to the count
                        dict_nodes[(nodei,nodej)] = dict_nodes[(nodei,nodej)] + 1

        return dict_nodes             

    def calculate_k(self, communities: list, statistic = True):

        '''
        This function is for calculate the k of a community.

        Parameters
        ----------        
        communities : list
            A list of comunities.
        statistic : bool
            If True, the function will calculate the k of all communities with the statistic method.
        
        Returns
        -------
        result : int
            The k of the community.
        '''

        k = 0
        data = []
        if statistic:
            for community in communities:
                for partition in community:
                    data.append(len(partition))

            mean = statistics.mean(data)
            stddev = statistics.stdev(data)    

            

            count = 0
            for x in data:
                if x in range(mean - (-2 * stddev) , mean + (2 * stddev)):
                    count += 1

            
            k = count/len(communities)
            return k

    
    def merge_communities_dict(self, general_dict: dict, dict_new: dict):
    
        '''
        Merges two dictionaries into a single dictionary.

        Args:
        -----
            general_dict (dict): General dictionary where the values of the new dictionary will be added.
            dict_new (dict): New dictionary with values that will be added to the general dictionary.
        '''

        # For each key in the new dictionary:
        for key in dict_new:
            # If the key is not in the general dictionary
            if key not in general_dict:
                # Add it to the general dictionary
                general_dict[key] = dict_new[key]
            # If the key is in the general dictionary
            else:
                # Add the value from the new dictionary to the value in the general dictionary
                general_dict[key] = general_dict[key] + dict_new[key]

    def important_nodes(self):

        '''
        This function is for calculate the important nodes of the graph.

        Returns
        -------
        result : dict
            A dict with the important nodes.
        '''

       

        dict_degree = dict(self.G.degree()) # type: ignore

        dict_weighted = dict(self.G.degree(weight='weight')) # type: ignore

        list_degree = list(dict_degree.items())

        list_weighted = list(dict_weighted.items())

        list_degree.sort(key=lambda x: x[1], reverse=True)

        list_weighted.sort(key=lambda x: x[1], reverse=True)


        list_degree = list_degree[:7200]

        list_weighted = list_weighted[:7200]

        list_degree = [x[0] for x in list_degree]

        list_weighted = [x[0] for x in list_weighted]

        intersection = set(list_degree).intersection(set(list_weighted))

        return intersection

        
    
    # Begining of Horacio's Region

    def edgeWithinComm(self, vi, si, w, directed):
        sugSi = nx.subgraph(self.G, si) # type: ignore
        if not directed:
            ki = sugSi.degree(vi, weight = w)
        else:
            ki = sugSi.out_degree(vi, weight = w)
        return ki

    def degMeanByCommunity(self, si, w, directed):
        sum = 0
        for vi in si:
            edgeWith = self.edgeWithinComm(vi, si, w, directed)
            sum += edgeWith
        return sum/len(si)

    def standDesvCommunity(self, si, w, directed):
        dMeanCi = self.degMeanByCommunity(si, w, directed)
        sum = 0 
        for vi in si:
            ki = self.edgeWithinComm(vi, si, w, directed)
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
                
        

        
    def update_graph_with_within_measure(self, communities ,algorithms = ['louvain', 'greedy', 'lpa', 'infomap']):

        for algorithm in algorithms:
                

            wdnw = pickle.load(open('dataset/outputs/FlyCircuitResult/within_'+ algorithm + '_directed_notWeighted', 'rb'))
            wdw = pickle.load(open('dataset/outputs/FlyCircuitResult/within_'+ algorithm + '_directed_weighted', 'rb'))
            wunw = pickle.load(open('dataset/outputs/FlyCircuitResult/within_'+ algorithm + '_notDirected_notWeighted', 'rb'))
            wuw = pickle.load(open('dataset/outputs/FlyCircuitResult/within_'+ algorithm + '_notDirected_weighted', 'rb'))

            files = [(wdnw, 'withing_directed_notWeighted'), (wdw, 'withing_directed_Weighted'), (wunw, 'withing_notDirected_notWeighted'), (wuw,  'withing_notDirected_Weighted')]

            for data, name in files:
                for k, v in data.items():
                    
                    for i in range(len(v)):

                        community = communities[i]
                        community_number = 0

                        if algorithm == 'infomap':
                            for j in range(len(community['communities'])):                        
                                if k in community['communities'][j]:
                                    community_number = j
                                    break
                        else:
                            for j in range(len(community)):                        
                                if k in community[j]:
                                    community_number = j
                                    break
                            
                        key = (algorithm, str(i), str(community_number))
                        if key not in self.G.nodes[k]['data'].keys():
                            self.G.nodes[k]['data'][(algorithm, str(i), str(community_number))] = {name: v[i]} # type: ignore
                        else:
                            self.G.nodes[k]['data'][(algorithm, str(i), str(community_number))][name] = v[i]
            

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
        
    def save_attributed_graph_to_csv(self, path: str = 'dataset/outputs/attributed_graph.csv'):

        columns = ['id', 'degree', 'in_degree', 'out_degree', 'weight', 'in_weight', 'out_weight']

        columns.extend(self.G.nodes['104198-F-000000'].keys())

        columns.remove('data')

        keys = self.G.nodes['104198-F-000000']['data'].keys()

        for alg, iter, ci in keys:
            columns.append(alg + '_' + iter + '_' + 'ci')

        for alg, iter, ci in keys:
            columns.append(alg + '_' + iter + '_' + 'ci' + '_participation_coefficient')
            columns.append(alg + '_' + iter + '_' + 'ci' + '_whiting_directed_weighted')
            columns.append(alg + '_' + iter + '_' + 'ci' + '_whiting_directed_notweighted')
            columns.append(alg + '_' + iter + '_' + 'ci' + '_whiting_notdirected_weighted')
            columns.append(alg + '_' + iter + '_' + 'ci' + '_whiting_notdirected_notweighted')
            

        df = pd.DataFrame(columns=columns)

        for node in self.G.nodes():

            data  = {'id': node,  'degree': self.G.degree(node), 'in_degree': self.G.in_degree(node), 'out_degree': self.G.out_degree(node),  # type: ignore
                            'weight': self.G.degree(node, weight='weight'), 'in_weight': self.G.in_degree(node, weight='weight'), 'out_weight': self.G.out_degree(node, weight='weight'),  # type: ignore
                            'eigenvector_centrality': self.G.nodes[node]['eigenvector_centrality'], 'eigenvector_centrality_weighted': self.G.nodes[node]['eigenvector_centrality_weighted'],
                            'pagerank': self.G.nodes[node]['pagerank'], 'degree_centrality': self.G.nodes[node]['degree_centrality'], 'core_number': self.G.nodes[node]['core_number'],
                            'closeness_centrality': self.G.nodes[node]['closeness_centrality'], 'clustering_coefficient': self.G.nodes[node]['clustering_coefficient'], 
                            'vertex_betweenes': self.G.nodes[node]['vertex_betweenes']}
            
            for alg, iter, ci in self.G.nodes[node]['data'].keys():
                data[alg + '_' + iter + '_' + 'ci'] = ci
                data[alg + '_' + iter + '_' + 'ci' + '_participation_coefficient'] = self.G.nodes[node]['data'][(alg, iter, ci)]['participation_coefficient']
                data[alg + '_' + iter + '_' + 'ci' + '_whiting_directed_weighted'] = self.G.nodes[node]['data'][(alg, iter, ci)]['withing_directed_Weighted']
                data[alg + '_' + iter + '_' + 'ci' + '_whiting_directed_notweighted'] = self.G.nodes[node]['data'][(alg, iter, ci)]['withing_directed_notWeighted']
                data[alg + '_' + iter + '_' + 'ci' + '_whiting_notdirected_weighted'] = self.G.nodes[node]['data'][(alg, iter, ci)]['withing_notDirected_Weighted']
                data[alg + '_' + iter + '_' + 'ci' + '_whiting_notdirected_notweighted'] = self.G.nodes[node]['data'][(alg, iter, ci)]['withing_notDirected_notWeighted']

            

            df.loc[len(df)] = data # type: ignore

        df.to_csv(path, index=False)

        

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

def apply_central_dispersion_measures(path: str):

    data = pd.read_csv('output/attributed_graph-1.4.1.csv', header=0)

    #data = data[[' weight', ' edge_betweenness']]

    data = data[['degree', 'in_degree', 'out_degree', 'weight', 'in_weight', 'out_weight', 'eigenvector_centrality', 'eigenvector_centrality_weighted', 'pagerank', 'degree_centrality', 'core_number', 'closeness_centrality', 'clustering_coefficient', 'vertex_betweenes']]

    #Calculate summary statistics
    summary = data.describe()
    

    # Add mean, median, mode, variance, standard deviation, skewness, and kurtosis
    summary.loc['mean'] = data.mean()
    summary.loc['median'] = data.median()
    summary.loc['mode'] = data.mode().iloc[0]
    summary.loc['variance'] = data.var()    
    summary.loc['skewness'] = data.skew()
    summary.loc['kurtosis'] = data.kurtosis()

    sw_test_results = pd.DataFrame({'shapiro_w': []})
    for col in data.columns:
        sw, p = stats.shapiro(data[col])
        sw_test_results.loc[col] = [p]

   
    # Add the Shapiro-Wilk test results to the original DataFrame
    summary.loc['shapiro_w'] = sw_test_results['shapiro_w']

    # Export summary statistics to a CSV file
    summary.to_csv(path)



if __name__ == '__main__':
    

    m = Matrix([], {},[])
    
    m.load_matrix_obj(path='dataset/attributed_graph-1.4.fly')

    print(m.G.number_of_edges())

    print(datetime.datetime.now())
    
    m.G = nx.generators.social.karate_club_graph()
    
    list_of_communities = []

    for i in range(3):
        result = nx.algorithms.community.label_propagation.label_propagation_communities(m.G)
        list_of_communities.append([list(x) for x in result])
  
    for x in list_of_communities:
        print(x)
    
    print('\n')

    value = m.RoughClustering(communities=list_of_communities)

    for g in value:
        print(g.nodes)

    #data = pd.read_csv('output/attributed_graph-1.4.1.csv', header=0)

    #influen = m.important_nodes()

    #G = m.G.subgraph(inter)

    #kvalue = nx.algorithms.core.core_number(G)
    
    #influen = set(influen)

    #kvalue = pickle.load(open('output/kvalue.pkl', 'rb'))

    # kvalue = nx.algorithms.core.core_number(m.G)

    # data_dict = {}
    # data_node = {}

    # for key, value in kvalue.items():
    #     if value not in data_dict:
    #         data_dict[value] = 1
    #         keys = set()
    #         keys.add(key)
    #         data_node[value] = keys
    #     else:
    #         data_dict[value] += 1
    #         data_node[value].add(key)

    # data  =  list(data_dict.items())
    # data.sort(key=lambda x: x[1], reverse=True)
    
    # k = 1
    # N = len(influen)
    # for key, value in data:
    #     influen = influen.difference(data_node[key])        
    #     if len(influen) < (N * 0.2):
    #         break
    #     k += 1

    # print(k)
    # list_kvalue = list(kvalue.items())

    # list_kvalue.sort(key=lambda x: x[1], reverse=True)

    

    # set_inter = set(inter)
    

    # print(len(set(list_kvalue)))
    

   
            

        
   




   

    #data = pd.read_csv('output/edgesMeasure.csv', header=0)

   
    # Plot a histogram of the data in a three ranges
    #plt.hist(data['degree'], bins=3, range=(0, 4039)) 
    # plt.hist(data['degree'], bins=20)
    # plt.show()
    
    # plt.boxplot(data['degree'], meanline=True, showmeans=True, medianprops=dict(color='yellow'))
    # plt.show()

    # Plot a Q-Q plot to check if the data is normally distributed
    # stats.probplot(data['degree'], dist="norm", plot=plt, )
    # plt.show() 
    
    
    print(datetime.datetime.now())
    
    
    

   