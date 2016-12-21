'''
Tensorflow implementation of KMeans++
@author: Utkarsh Simha (usimha@ucsd.edu)
'''

import tensorflow as tf
import numpy as np
import scipy.spatial
from sklearn import datasets
import tensorflow as tf
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

    
class KMeansPP:
    def __init__( self, n_clusters, n_iters=10 ):
        self.graph = tf.Graph()
        self.sess = tf.Session( graph = self.graph )
        self.n_clusters = n_clusters
        self.n_iters = n_iters

    def fit( self, X ):
        ''' 
            -- DESCRIPTION --
            Initializes the centroids and assigns each data point into  a cluster
            Iteratively updates the centroids to the mean of each cluster 

            -- PARAMS --
            * X : numpy.ndarray, shape (n_samples, n_features)
                  Training data

            -- RETURNS --
            None 
        '''
        n_samples = X.shape[0]
        dim = X.shape[1]
        with self.graph.as_default():
            self.train_x = tf.constant( X, tf.float64, name='train_x' )
            centroids = self._initCentroids( X )
            self.centroids = tf.Variable( centroids, dtype=tf.float64, name='centroids' )
            cluster_assign = tf.Variable( tf.zeros( [ n_samples ], dtype=tf.int32, name='cluster_assign' ) )
            expanded_centroids = tf.expand_dims( self.centroids, 1 )
            expanded_data = tf.expand_dims( self.train_x, 0 )
            local_dist = tf.reduce_sum( tf.square( expanded_centroids - expanded_data ), 2 )
            self.cent_assignments = tf.argmin( local_dist, 0 )
            self.means = []
            for c in range( self.n_clusters ):
                mean = tf.reduce_mean(
                        tf.gather( self.train_x, tf.reshape( tf.where(
                            tf.equal( self.cent_assignments, c )
                            ), [1,-1] ) ), 1
                        )
                self.means.append( mean )

            recalc_centroids = tf.concat( 0, self.means )
            self.update_centroids = tf.assign( self.centroids, recalc_centroids )
            tf.initialize_all_variables().run(session=self.sess)
        for i in range( self.n_iters ):
            self.sess.run( self.cent_assignments )
        self.labels_ = self.sess.run( self.cent_assignments )

    def predict( self, X ):
        ''' 
            -- DESCRIPTION --
            Returns the cluster assignments for each data point
            using the fitted centroids 

            -- PARAMS --
            * X : numpy.ndarray, shape (n_samples, n_features)
                  Testing data

            -- RETURNS --
            cent_assignments : numpy.ndarray, shape ( n_samples, 1 )
        '''
        n_samples = X.shape[0]
        dim = X.shape[1]
        with self.graph.as_default():
            test_x = tf.constant( X, tf.float64 )
            cluster_assign = tf.Variable( tf.zeros( [ n_samples ], dtype=tf.int32 ) )
            expanded_centroids = tf.expand_dims( self.centroids, 1 )
            expanded_data = tf.expand_dims( test_x, 0 )
            local_dist = tf.reduce_sum( tf.square( expanded_centroids - expanded_data ), 2 )
            cent_assignments = tf.argmin( local_dist, 0 )
            tf.initialize_all_variables().run(session=self.sess)
        return self.sess.run( cent_assignments )


    def _initCentroids( self, train_x,  random=False ):
        ''' 
            -- DESCRIPTION --
            Initializes centroids with probability proportional to the 
            distances between the data points and the chosen centroids 

            -- PARAMS --
            * train_x : numpy.ndarray, shape (n_samples, n_features)
                        Training data
            * random : bool
                       If the centroids should be chosen randomly

            -- RETURNS --
            centroids : numpy.ndarray, shape ( n_clusters, n_features )
                        The centroids initialized for each cluster
        '''
        if random:
            return np.random.choice( train_x, self.n_clusters )
	centroids = np.zeros( ( self.n_clusters, train_x.shape[1] ) )
	centroids[ 0 ] = train_x[ np.random.random_integers( 0, train_x.shape[0] ) ] #Randomly choose one data point 
	expanded_data = np.expand_dims( train_x, 0 )
	k = 1
	while( k < self.n_clusters ):
	    expanded_centroids = np.expand_dims( centroids, 1 )
	    D_x = np.min( np.sum( np.square( expanded_centroids - expanded_data ), 2 ), 0 )
	    probs = D_x/float(np.sum( D_x ) )
	    cumprobs = np.cumsum( probs )
	    rand_prob = np.random.random()
	    index = np.where( rand_prob < cumprobs )[0][0]
	    centroids[k] = train_x[ index ]
	    k+=1 
	return centroids 


if __name__ == '__main__':
    NUM_CENTROIDS = 10
    digits = datasets.load_digits()
    train_x, train_y = digits.data, digits.target
    n_iters = 10
    clust = KMeansPP( NUM_CENTROIDS, n_iters )
    clust.fit( train_x )
