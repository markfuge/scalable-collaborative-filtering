'''Testing a scalable collaborative filtering algorithm that implements
   stochastic gradient descent on a regularized loss function over user
   preferences for movies in the MovieLens 10M collection.'''
import sys
from collab_filter import *
from movie_lens import *

if __name__ == "__main__":    
    # Import and process the data in the MovieLens module

    if len(sys.argv)>2:
        test_categories =True
    else:
        test_categories =False
    cat_str = 'with' if test_categories else 'without'
    print "Running "+cat_str+' category attributes'

    print "Retrieving data file..."
    movie_attributes = return_movie_attributes()
    training,testing,num_users,num_movies = get_data()
    #training= training[-int(len(training)*0.1):]
    print "Data retrieved. # Training data: "+str(len(training))+", # Testing: "+str(len(testing))

    cf=CollaborativeFilter(num_users,num_movies,categories=test_categories)
    if len(sys.argv)>1:
        cf.Lambda = float(sys.argv[1])
        
    n_avg = 100000
    print "Lambda: "+str(cf.Lambda)
    iterfile = open(str(cf.Lambda)+'-'+cat_str+'iterlog.txt','w')
    comparefile = open(str(cf.Lambda)+'-'+cat_str+'-comparelog.txt','w')
    movie_error=dict()
    movie_counts=dict()
    for m in range(0,10):  # iterations through the dataset
        total_loss=0
        print "Starting iteration "+str(m+1) +" through the dataset."
        average_loss=0
        for i,line in enumerate(training):
            time,userid,movieid,rating = line
            attrs = movie_attributes[movieid]['categories']
            loss=cf.update(userid,movieid,rating,attrs)
            total_loss+=loss
            average_loss+=loss
            if i % n_avg ==0:
                iters = 'Iter: %d; Average Loss: %.2e; Eta: %.2e' % (i,average_loss/n_avg,cf.eta(cf.iteration))
                print iters
                iterfile.write(iters)
                average_loss=0
        print 'Average Training Loss: '+str(total_loss/len(training))

        # Now time to do testing
        training_error=0
        for line in training:
            time,userid,movieid,rating = line
            attrs = movie_attributes[movieid]['categories']
            error=cf.predict(userid,movieid,rating,attrs)
            training_error+=error
        testing_error =0
        movie_error=dict()
        for line in testing:
            time,userid,movieid,rating = line
            attrs = movie_attributes[movieid]['categories']
            error=cf.predict(userid,movieid,rating,attrs)
            testing_error+=error
            if movieid not in movie_error:
                movie_error[movieid] =0
                movie_counts[movieid] = 0
            movie_error[movieid] += error
            movie_counts[movieid] += 1
        compare_str='Lambda: %f; Iterations: %d; Avg. Training error: %e; Avg. Testing error: %e' % (cf.Lambda,cf.iteration,training_error/len(training),testing_error/len(testing))
        print compare_str
        comparefile.write(compare_str)
        
    # Save the model for later
    cf.save_model()
    movie_err_file = open(str(cf.Lambda)+'-'+cat_str+'-movie_err.txt','w')
    reverse_map = [item[0] for item in sorted(category_map.items(),key=lambda tup: tup[1])]
    for id,error in movie_error.iteritems():
        s = '%d;%s;%.4f;' % (id,movie_attributes[id]['title'],error/movie_counts[id])
        for attr in movie_attributes[id]['categories']:
            s+= '%s;' % reverse_map[attr]
        s+='\n'
        movie_err_file.write(s)
    movie_err_file.close()
    # Close some logging files
    iterfile.close()
    comparefile.close()

