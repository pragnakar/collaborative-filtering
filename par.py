from multiprocessing import Pool

def f(i):
    print(i)
    return i*i

#if __name__ == '__main__':
#    p = Pool(5)
 #   print(p.map(f, [1, 2, 3]))
  #  print('start-')



import  numpy as np
import pandas as pd
from math import sqrt
data_train = pd.read_csv('data/ratings_training_80.csv') # modify to get the path
data_test = pd.read_csv('data/ratings_test_20.csv').ix[:500,:]

data_train_v1 = data_train.copy()

for i in range(0,len(data_train_v1.user_id.unique())):
    #print(i)
    data_train_v1.ix[data_train_v1.user_id == i,'ratings'] = data_train_v1.ix[data_train_v1.user_id == i ]['ratings'] - np.average(data_train_v1.ix[data_train_v1.user_id == i ].ratings.values)
rating_matrix= data_train_v1.pivot(index='user_id',columns='movie_id', values='ratings').fillna(value=0)

# cosine distance


def cosine_distance( i,j):
    temp=rating_matrix.ix[(rating_matrix.ix[:,i]!=0)& (rating_matrix.ix[:,j]!=0)].ix[:,(i,j)]
    try:
        return (np.sum(temp.ix[:,i] * temp.ix[:,j])) /( (sqrt(np.sum((temp.ix[:,i])**2))) * (sqrt(np.sum((temp.ix[:,j])**2))) )
    except ZeroDivisionError:
        return 0


similarity_matrix = pd.DataFrame(np.zeros((9216, 9216)))
similarity_matrix = similarity_matrix.replace(0,-2)

from multiprocessing import process


predicted_val_list=[]
rar= range(0,data_test.shape[0])
def fun(i):
    target_user_id = data_test.ix[i,0]
    target_movie_id = data_test.ix[i,1]
    target_rating = data_test.ix[i,2]
    list_target_watched = data_train.ix[data_train.user_id == target_user_id].movie_id.values
    movies_watched = data_train.ix[data_train.user_id == target_user_id]
    cosine_distance_list =[]
    for j in list_target_watched:
        temp=similarity_matrix.ix[target_movie_id,j]
        if  temp != -2:
            cosine_distance_list.append(temp)
        else:
            temp=similarity_matrix.ix[j, target_movie_id]=similarity_matrix.ix[target_movie_id,j] = cosine_distance(target_movie_id,j)
            cosine_distance_list.append(temp)
    movies_watched['cosine_distance'] = pd.Series(cosine_distance_list, index=movies_watched.index)
    ########################## patch # what do if we have empty set
    try:
        sub_set =movies_watched.ix[movies_watched.cosine_distance>0]
        predicted_value = round(np.sum(sub_set.cosine_distance * sub_set.ratings)/np.sum(sub_set.cosine_distance),1)
    except ZeroDivisionError:
        predicted_value = 0

    ###########################################


    #predicted_value = round(np.sum(sub_set.cosine_distance * sub_set.ratings)/np.sum(sub_set.cosine_distance),1)
    print(i,')',' actural rating', target_rating, ' predicted value : '  ,predicted_value)
    #predicted_val_list.append(predicted_value)
    return predicted_value

if __name__ == '__main__':
    import time
    start = time.time()
    p = Pool(5)
    predicted_val_list=p.map(fun, rar)
    data_test['predicted_value'] = pd.Series(predicted_val_list, index=data_test.index)
    data_test.to_csv('data/CF_output-par')
    end = time.time()
    print('run time ', end-start)
