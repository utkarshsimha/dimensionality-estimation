import cPickle as pickle
import numpy as np
import sklearn
from scipy.sparse import linalg as sparse_linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tf_kmeans_pp as kmeans_pp

def generate_data(n_samples, data_type = 'circles'):
	'''
		-- DESCRIPTION --
		Generate Synthetic dataset 

		-- PARAMS --
		* n_samples : integer
			Number of samples to generate
		* data_type : {default 'circles', 'swissroll', 'moons'} 
			Type of dataset to generate

		__ RETURNS --
		X : numpy.ndarray, shape (n_samples, n_features)
			Dataset with n_samples, each sample with n_features

	'''
	np.random.seed(0)
	if data_type == 'circles':	data = sklearn.datasets.make_circles(n_samples = n_samples, noise = 0.05, factor = .5) 
	elif data_type == 'swissroll':	data = sklearn.datasets.make_swiss_roll(n_samples = n_samples, noise = 0.5) 
	elif data_type == 'moons':	data = sklearn.datasets.make_moons(n_samples = n_samples, noise = 0.05)
	else:	raise ValueError( "Unsupported data_type {}".format(data_type) )
	X, Y = data
	return X

def plot( X, labels ):
	'''
		-- DESCRIPTION --
		Plot dataset clusters

		-- PARAMS --
		* X : numpy.ndarray, shape (n_samples, n_features)
			Dataset
		* labels : numpy.ndarray, shape (n_samples, 1)
			Cluster labels

	'''

	if X.shape[1] == 2:
        plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels )
    elif X.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter( X[:, 0], X[:, 1], X[:, 2], c=labels )
	elif:
		raise ValueError( "Unsupported dimension of data {}".format(X.shape[1]) ) 
    plt.show()

class SpectralClustering:
	'''
		-- DESCRIPTION --
		Apply Spectral Clustering using Graph Laplacian

	'''

    def __init__( self, n_components = 2, n_neighbors = 12, affinity = 'knn', laplacian = 'norm_sym',
				  clustering = 'kmeans++', dist_fn = lambda x, y: np.linalg.norm(x - y),
				  weight_fn = lambda x: 1 if x >= 0.5 else 0 ):
     ''' 
	 	-- PARAMS -- 
		* n_components : integer, optional
			No. of clusters or connected graph components
		* n_neighbors : integer, optional
			No. of k-nearest neighbors, if knn similarity measure is used
		* affinity : {default 'knn', 'rbf', 'heat_kernel'}
			Method to compute affinity graph/similarity matrix
		* laplacian : {'unnorm', default 'norm_sym', 'norm_rw'}
			Type of Graph Laplacian (Unnormalized,`Normalized - symmetric, random walk)
		* clustering : {default 'kmeans++', 'kmeans'}
			Type of centroid-based clustering
		* dist_fn : callable, optional, default - Euclidean distance
			Callable that returns distance between two points
		* weight_fn : callable, optional, default - 0/1 weight function with 0.5 threshold
			Callable that returns weight given a distance between two points 
	'''
	 	self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.laplacian = laplacian
        self.clustering = clustering 
        self.dist_fn = dist_fn
        self.weight_fn = weight_fn

    def fit( self, X ):
		'''
			-- DESCRIPTION --
			Builds the affinity graph/similarity matrix using a distance and weight function
			Computes Graph Laplacian of the affinity graph 
			Obtains spectral embedding using leading eigenvectors of the Graph Laplacian
			Clusters embeddings using centroid-based clustering algorithms
			Spectral Analysis of projection of embeddings onto leading eigenvectors

			-- PARAMS --
			* X : numpy.ndarray, shape (n_samples, n_features)
				Set of data for spectral clustering

		'''
        self.X = X
        self._build_affinity_graph()
        self._compute_graph_laplacian()
        self._compute_embedding()
        self._clustering()
        self._project_plot()

    def _build_affinity_graph( self ):
        '''
			-- DESCRIPTION --
			Compute graph of affinities between instances in the dataset using the distance function

		'''
        dist = np.zeros( ( len(self.X), len(self.X) ) )
        for i in range(len(self.X)):
            for j in range(len(X)):
                dist[i][j] = self.dist_fn( X[i], X[j] )

		if self.affinity == 'rbf':
			gamma = 1 / float(X.shape[0])
			self.W = np.exp(-gamma * np.square(dist) ) 
		elif self.affinity == 'heat_kernel':
			t = 15 
			eps = 0.01
			self.W = np.zeros((self.X.shape[0], self.X.shape[0]))
			self.W[np.where(dist > eps)] = np.exp(-dist[np.where(dist > eps)] / (4 * t))
		elif self.affinity == 'knn':
			dist[np.diag_indices_from(dist)] = np.inf
			idx = np.argsort(dist, axis = 1)
			self.W = np.zeros(dist.shape)	
			for i in range(self.X.shape[0]):
				for j in range(i):
					if j in idx[i, :self.n_neighbors]:
						self.W[i][j] = self.W[j][i] = self.weight_fn( dist[i][j] )
		else:
			raise ValueError( "Unsupported affinity type: {}".format( self.affinity ) )

		d = np.sum(self.W, axis = 1)
		self.D = np.diag( d )

    def _compute_graph_laplacian( self ):
        '''
			-- DESCRIPTION --
			Compute normalized/unnormalized laplacian of affinity graph
		'''
        if self.laplacian == 'norm_sym':
            D_minus_half = scipy.linalg.fractional_matrix_power(np.linalg.pinv(self.D), 0.5) 
            self.L = np.identity(self.W.shape[0]) - np.dot( np.dot( D_minus_half, self.W ), D_minus_half )
        elif self.laplacian == 'norm_rw':
            self.L = self.D - self.W
        elif self.laplacian == 'unnorm':
            self.L = self.D - self.W
        else:
            raise ValueError( "Unsupported laplacian type: {}".format( self.laplacian ) )

    def _compute_embedding( self ):
        '''
			-- DESCRIPTION --
			Compute spectral embeddings of the dataset by solving the Laplacian eigenproblem 
			(Find the leading eigenvectors of the Laplacian as embeddings) 	

		'''

        K = self.n_components * 10
		try:
			if self.laplacian == 'norm_sym':
				eigenvalues, eigenvectors = sparse_linalg.eigsh(self.L, K, M = None, which = 'SM', sigma = None) 
			elif self.laplacian == 'norm_rw':
				eigenvalues, eigenvectors = sparse_linalg.eigsh(self.L, K, M = self.D, which = 'SM', sigma = None) 
			elif self.laplacian == 'unnorm':
				eigenvalues, eigenvectors = sparse_linalg.eigsh(self.L, K, which = 'SM', sigma = None)
		except Exception as e:
				eigenvalues, eigenvectors = e.eigenvalues, e.eigenvectors
		if self.laplacian == 'norm_sym':
					eigenvectors = sklearn.preprocessing.normalize( eigenvectors, axis=0 )
			self.eigenvalues = eigenvalues
			self.eigenvectors = eigenvectors
			self.embedding = self.eigenvectors[ :, :self.n_components ]

    def _clustering( self ):
		'''
			-- DECRIPTION --
			Cluster spectral embeddings using centroid-based clustering algorithms 
			Obtain cluster labels for instances in the dataset

		'''

		if self.clustering == 'kmeans':
			kmeans = sklearn.cluster.KMeans( self.n_components )
			kmeans.fit( self.embedding )
			self.labels = kmeans.labels_
		elif self.clustering == 'kmeans++':
			kmeans = kmeans_pp.KMeansPP( self.n_components )
			kmeans.fit( self.embedding )
			self.labels = kmeans.labels_
        else:
            raise ValueError( "Unsupported clustering method: {}".format( self.clustering ) )

    def _project_plot( self ):
		'''
			-- DESCRIPTION --
			Obtain color spectrum of projection of instances of dataset onto leading eigenvectors 

		'''

        projection = self.eigenvectors[:,1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scalarMap = matplotlib.cm.ScalarMappable()
        colorMap = scalarMap.to_rgba( projection )
        ax.scatter( self.X[:, 0], self.X[:,1], self.X[:, 2], c=colorMap, cmap=plt.cm.jet, edgecolor=colorMap )
        scalarMap.set_array( projection )
        plt.colorbar( scalarMap )
        plt.show()

if __name__ == '__main__':
	X = generate_data(n_samples = 2500, data_type = 'swissroll')
	sc = SpectralClustering( n_neighbors=20 , weight_fn = lambda x: np.exp(-x ** 2 / (4 * 5.0)) )
	sc.fit( X )
