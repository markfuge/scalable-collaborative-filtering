'''Helper module for loading and processing the MovieLens 10M data'''
import re
import os
from random import shuffle

DATA_PATH = '.\\data\\'
MOVIE_PATH = 'movies.dat'
RATING_PATH = 'ratings.dat'
mov_delim = re.compile('::')
import cPickle as pickle

def return_movie_attributes():
    '''Returns an associative array keyed by the movie id for faster lookup later.
       Parses a line in the MovieLens format - example:
       "3::Grumpier Old Men (1995)::Romance|Comedy\n"
       '''
    attr_delim = re.compile('\|')   # MovieLens categorie example: ::Drama|Fantasy|Comedy\n
    attributes={}
    with open(DATA_PATH+MOVIE_PATH,'r') as file:
        for line in file.readlines():
            id,movie_title,attribute_list = mov_delim.split(line.strip())
            id = int(id)
            attribute_list = attr_delim.split(attribute_list)
            attributes[id]=dict()
            attributes[id]['title']=movie_title.decode('utf-8')
            attributes[id]['categories']=attribute_list
    return attributes

def return_ratings():
    '''Returns an array of movie ratings and timestamps'''
    with open(DATA_PATH+RATING_PATH,'r') as file:
        ratings=[]
        for line in file.readlines():
            user_id,movie_id,rating,timestamp = mov_delim.split(line.strip())
            ratings.append((int(timestamp),int(user_id),int(movie_id),float(rating)))
    return ratings

def split_ratings(ratings):
    '''Splits the ratings into validation and test sets.
       To do this, we first sort the ratings by timestamp, and then
       split off the later 20% of the data.'''
    N=len(ratings)
    cutoff=int(round(N*.8))
    ratings.sort()
    return ratings[:cutoff],ratings[cutoff:]

def clean_testing(validation,testing):
    validation_movie_id_set=set()
    for item in validation:
        validation_movie_id_set.add(item[2])    # item[2] is the movie id

    for i,item in enumerate(testing):
        if item[2] not in validation_movie_id_set:
            testing.pop(i) # Then we should delete this entry
    return testing

def get_user_movie_stats(data):
    user_set = set()
    movie_set = set()
    for item in data:
        user_id = item[1]
        movie_id = item[2]
        if user_id not in user_set:
            user_set.add(user_id)
        if movie_id not in movie_set:
            movie_set.add(movie_id)
    return len(user_set),len(movie_set)

def data_exists():
    '''Checks if the validation and testing data file exist already - returns True if they do.'''
    return os.path.exists('validation.dat') and os.path.exists('testing.dat') and os.path.exists('nums.dat')


def get_data():
    '''Retrieves the relevant data from disk and does any necessary pre-processing. Specifically:
       1) Reads in the movie and ratings files.
       2) splits the data set into validation and testing.
       3) Discards movies/ratings that only appear in the test set    
    '''
    if not data_exists():
        validation,testing = split_ratings( return_ratings() )
        # Only include test movies that were in the database
        testing = clean_testing(validation, testing)
        # Permute the order of the test set
        shuffle(testing)
        num_users,num_movies = get_user_movie_stats(validation)
        # Dump the files for later
        pickle.dump(validation,file('validation.dat','wb'))
        pickle.dump(testing,file('testing.dat','wb'))
        pickle.dump((num_users,num_movies),file('nums.dat','wb'))
    else:
        print 'loading existing files'
        validation = pickle.load(file('validation.dat','rb'))
        testing = pickle.load(file('testing.dat','rb'))
        num_users,num_movies = pickle.load(file('nums.dat','rb'))

    return validation,testing,num_users,num_movies

