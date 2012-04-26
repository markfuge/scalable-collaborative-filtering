import numpy as np
import math

def rho(bu,bm,nu_u,nu_m,rating):
    return (bu+bm+nu_u.dot(nu_m) - rating)

def adjust_eta(initial_eta):
    return lambda t: initial_eta/(math.sqrt(t))

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

        # Initialize the latent factor space for users and movies
        # Random initialization
        self.nu_u = dict()
        self.nu_m = dict()
        # Intialize the learning rate for stochastic gradient descent
        initial_eta=0.1
        self.eta = adjust_eta(initial_eta)
        self.iteration=0
        self.Lambda = 0

    def update(self,user_id,movie_id,rating):
        # Determine our descent step size
        self.iteration+=1
        eta = self.eta(self.iteration)
        # Fetch the relevant vectors
        nu_u,bu = self.get_user(user_id)
        nu_m,bm = self.get_movie(movie_id)

        discount = 1-self.Lambda*eta
        nu_u = discount*nu_u - eta*nu_m*rho(bu,bm,nu_u,nu_m,rating)
        nu_m = discount*nu_m - eta*nu_u*rho(bu,bm,nu_u,nu_m,rating)
        bu = discount*bu - eta*rho(bu,bm,nu_u,nu_m,rating)
        bm = discount*bm - eta*rho(bu,bm,nu_u,nu_m,rating)
        self.bm[movie_id] = bm
        self.bu[user_id] = bu
        self.nu_u[user_id] = nu_u
        self.nu_m[movie_id] = nu_m
        return self.loss(bu,bm,nu_u,nu_m,rating)
    
    def loss(self,bu,bm,nu_u,nu_m,rating):
       return 0.5*rho(bu,bm,nu_u,nu_m,rating)**2 + self.Lambda/2*(np.linalg.norm(nu_u)**2 + bu**2 + np.linalg.norm(nu_m)**2 + bm**2)

    def init_latent_factor_vector(self,dimensions):
        return np.random.rand(dimensions)-0.5

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

    def get_user(self,userid):
        '''Returns the nu and b vectors for a specific user id'''
        return (self.get(self.nu_u,userid,self.init_latent_factor_vector(self.num_latent)),
                self.get(self.bu,userid,np.random.rand()-.5))
    
    def get_movie(self,movieid):
        '''Returns the nu and b vectors for a specific movie id'''
        return (self.get(self.nu_m,movieid,self.init_latent_factor_vector(self.num_latent)),
                self.get(self.bm,movieid,np.random.rand()-.5))
