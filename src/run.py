#general imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

#import from src
from models import ContentRecommender
from models import CollabRecommender


def create_user_prof(beers):
    '''
    Takes inputs from user and creates a dictionary containing favorite beers.
    
    Inputs:
        df - user dataframe
        beers - list of beers_ids chosen by user
        
    Returns:
        d - a dictionary containing users selected beers
    '''
    d={}
    for b in beers:
        d[b]=5
    return d



def similar_users(uid, df, n=10,thresh=4):
    '''
    Based on current user rating, returns most similar users that rated one or more of the same beers.
    
    Inputs:
        uid - user id of new user
        df - user-ratings dataframe updated with new user
        n - number of predictions to return
        thresh - threshold of number of rated beers in common for calculating similarities
    Returns:
        sim_users - array of most similar users ids
    '''
    user = df.loc[uid]
    ratings = (~user.isna())
    users_filt = df.T[ratings.values]
    users_filt = users_filt.T.dropna(thresh=thresh)
    #create pearson corr matrix
    U = np.corrcoef(users_filt)
    #find most similar users
    sim_users = np.argsort(U[0][~np.isnan(U[0])])[-(n+1):-1]
    return sim_users

if __name__ == '__main__':
    #import dataframes
    print('Importing top beers...')
    beer_df = pd.read_csv('data/top_beers.csv') #dataframe containing top 1000 beers
    
    ratings_df = pd.read_csv('data/top_ratings.csv') #dataframe containing reviews for top beers
    ratings_df = ratings_df[~(ratings_df.score>5)]  #filter outlier
    ratings_df['uid'] = ratings_df.groupby('username').ngroup() #create user ids for users
    
    beer = input("Enter a beer you like: ")
    brewery = input("What brewery is it from?: ")
    
    ####Content Recommender####
    content_r = ContentRecommender(beer_df)
    # content_r.fit()
    beer_recs = content_r.get_recommendations(item=beer,brewery=brewery)
    
    #create user dataframe to calculate user-user similarities
    user_df = ratings_df.pivot(index='uid',columns='beer_id',values='score')
    beers = [5,187317,203,17]
    d=create_user_prof(beers)
    uid = user_df.shape[0]
    updated_df = user_df.append(pd.DataFrame(d,index=[uid]))
    
    #find most similar users
    sim_users = similar_users(uid, updated_df,thresh=1)

    ####Collaborative Filter Recommender###
    colab_r = CollabRecommender(ratings_df)
    # colab_r.fit(model=SVD, n_factors=100, n_epochs=5, lr_all=0.01, reg_all=0.02)
    # colab_r.predict()
    top_ratings = colab_r.get_top_n()

    beer_dict={}
    for user in sim_users:
        for item in top_ratings[user]:
            if len(item)>0:
                beer_dict[item[0]]=item[1]
    beer_dict = dict(sorted(beer_dict.items(),key=lambda item: item[1], reverse=True))
    print(beer_dict)

