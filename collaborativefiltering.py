'''Testing a scalable collaborative filtering algorithm that implements
   stochastic gradient descent on a regularized loss function over user
   preferences for movies in the MovieLens 10M collection.'''

from collab_filter import *
from movie_lens import *

# Import and process the data in the MovieLens module
validation,testing,num_users,num_movies = get_data()

cf=CollaborativeFilter(num_users,num_movies)
for m in range(0,20):  # 10 iterations through the dataset
    total_loss=0
    for line in validation:
        time,userid,movieid,rating = line
        loss=cf.update(userid,movieid,rating)
        total_loss+=loss
        #print 'Loss: '+str(loss)
    print 'Total Loss: '+str(total_loss)
