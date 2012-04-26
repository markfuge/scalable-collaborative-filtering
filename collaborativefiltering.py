'''Testing a scalable collaborative filtering algorithm that implements
   stochastic gradient descent on a regularized loss function over user
   preferences for movies in the MovieLens 10M collection.'''

from collab_filter import *
from movie_lens import *

# Import and process the data in the MovieLens module
print "Retrieving data file..."
validation,testing,num_users,num_movies = get_data()
print "Data retrieved. # Validation data: "+str(len(validation))+", # Testing: "+str(len(testing))

cf=CollaborativeFilter(num_users,num_movies)
for m in range(0,20):  # 10 iterations through the dataset
    total_loss=0
    print "Starting iteration "+str(m+1) +" through the dataset."
    for line in validation:
        time,userid,movieid,rating = line
        loss=cf.update(userid,movieid,rating)
        total_loss+=loss
        #print 'Loss: '+str(loss)
    print 'Total Loss: '+str(total_loss)
