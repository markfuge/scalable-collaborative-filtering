import numpy as np
import math
import cPickle as pickle

def rho(mu,bu,bm,nu_u,nu_m,rating):
    return (mu+bu+bm+nu_u.dot(nu_m) - rating)

def adjust_eta(alpha,beta):
    return lambda t: 1/(math.sqrt(alpha+beta*t))

class CollaborativeFilter(object):
    """A collaborative filter object for storing and operating over the various
       parameters and the loss function"""

    def __init__ (self, num_users,num_movies,num_latent=50):
        '''Initializes the collaborative filtering model.
           Takes as arguments the number of users, the number of movies, and
           the amount of latent factors to model in the filter. The latent space
           defaults to 50 dimensions to keep memory reasonable.'''
        # Initialize the bias corrections for movies and users, respectively.
        # Random initialization
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_latent = num_latent
        self.bm = dict()
        self.bu = dict()
        self.mu=3

        # Initialize the latent factor space for users and movies
        # Random initialization
        self.nu_u = dict()
        self.nu_m = dict()
        
        # Intialize the learning rate for stochastic gradient descent
        self.initial_eta=0.1
        self.alpha = 10
        self.beta =0.01
        self.eta = adjust_eta(self.alpha,self.beta)
        self.iteration=0
        self.Lambda = 1

    def rho(self,rating):
        rho(self.mu,bu,bm,nu_u,nu_m,rating)

    def update(self,user_id,movie_id,rating):
        # Determine our descent step size
        self.iteration+=1
        eta = self.eta(self.iteration)
        #print "eta: "+str(eta)
        # Fetch the relevant vectors
        nu_u,bu = self.get_user(user_id)
        nu_m,bm = self.get_movie(movie_id)
        mu =self.mu

        discount = 1-self.Lambda*eta
        prediction = rho(mu,bu,bm,nu_u,nu_m,rating)
        nu_u = discount*nu_u - eta*nu_m*prediction
        prediction = rho(mu,bu,bm,nu_u,nu_m,rating)
        nu_m = discount*nu_m - eta*nu_u*prediction
        prediction = rho(mu,bu,bm,nu_u,nu_m,rating)
        bu = discount*bu - eta*prediction
        prediction = rho(mu,bu,bm,nu_u,nu_m,rating)
        bm = discount*bm - eta*prediction
        prediction = rho(mu,bu,bm,nu_u,nu_m,rating)
        mu = mu - eta*prediction
        self.mu=mu
        self.bm[movie_id] = bm
        self.bu[user_id] = bu
        self.nu_u[user_id] = nu_u
        self.nu_m[movie_id] = nu_m
        return self.loss(mu,bu,bm,nu_u,nu_m,rating)

    def predict(self,user_id,movie_id,rating):
        nu_u,bu = self.get_user(user_id)
        nu_m,bm = self.get_movie(movie_id)
        return rho(self.mu,bu,bm,nu_u,nu_m,rating)**2
    
    def loss(self,mu,bu,bm,nu_u,nu_m,rating):
        return 0.5*rho(mu,bu,bm,nu_u,nu_m,rating)**2 + self.Lambda/2*(nu_u.dot(nu_u) + bu**2 + nu_m.dot(nu_m) + bm**2)


    def get(self,dictionary,id,init_function):
        '''Fetches a value from a given dictionary or creates a new random latent factor vector
           and assigns it to the desired dictionary.'''
        if id in dictionary:
            return dictionary[id]
        else:
            # We have a new user we haven't seen before
            # lets initialize a random vector
            dictionary[id]= init_function
            return dictionary[id]

    def init_latent_factor_vector(self,dimensions):
        return np.random.rand(dimensions)-.5

    def get_user(self,userid):
        '''Returns the nu and b vectors for a specific user id'''
        return (self.get(self.nu_u,userid,self.init_latent_factor_vector(self.num_latent)),
                self.get(self.bu,userid,np.random.rand()-0.5))
    
    def get_movie(self,movieid):
        '''Returns the nu and b vectors for a specific movie id'''
        return (self.get(self.nu_m,movieid,self.init_latent_factor_vector(self.num_latent)),
                self.get(self.bm,movieid,np.random.rand()-0.5))

    def save_model(self):
        outfile = open(str(self.Lambda)+'-model.dat','wb')
        pickle.dump((self.bu,self.bm,self.nu_u,self.nu_m,self.initial_eta,self.iteration),outfile)
        outfile.close()
