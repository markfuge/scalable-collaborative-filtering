'''Testing a scalable collaborative filtering algorithm that implements
   stochastic gradient descent on a regularized loss function over user
   preferences for movies in the MovieLens 10M collection.'''
import sys
from collab_filter import *
from movie_lens import *

if __name__ == "__main__":    
    # Import and process the data in the MovieLens module
    print "Retrieving data file..."
    training,testing,num_users,num_movies = get_data()
    training= training[-int(len(training)*0.1):]
    print "Data retrieved. # Training data: "+str(len(training))+", # Testing: "+str(len(testing))

    cf=CollaborativeFilter(num_users,num_movies)
    if len(sys.argv)>1:
        cf.Lambda = float(sys.argv[1])
    n_avg = 100000
    print "Lambda: "+str(cf.Lambda)
    for m in range(0,10):  # 10 iterations through the dataset
        total_loss=0
        print "Starting iteration "+str(m+1) +" through the dataset."
        average_loss=0
        for i,line in enumerate(training):
            time,userid,movieid,rating = line
            loss=cf.update(userid,movieid,rating)
            total_loss+=loss
            average_loss+=loss
            if i % n_avg ==0:
                print 'Iter: '+str(i)+'; Average Loss: '+str(average_loss/n_avg)
                average_loss=0
        print 'Total Loss: '+str(total_loss)

        # Now time to do testing
        training_error=0
        for line in training:
            time,userid,movieid,rating = line
            error=cf.predict(userid,movieid,rating)
            training_error+=error
            #print 'Loss: '+str(loss)
        testing_error =0
        for line in testing:
            time,userid,movieid,rating = line
            error=cf.predict(userid,movieid,rating)
            testing_error+=error
            #print 'Loss: '+str(loss)
        print 'Lambda: '+ str(cf.Lambda) +'; Iterations: ' +str(cf.iteration)+'; Training error: '+str(training_error)+'; Testing error: '+str(testing_error)

    # Save the model for later
    cf.save_model()
