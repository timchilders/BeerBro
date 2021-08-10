#general imports
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import Image

#import from src
from models import ContentRecommender
from models import CollabRecommender

#streamlit app imports
import streamlit as st

@st.cache
def load_dataframes():
    print('Importing top beers...')
    beer_df = pd.read_csv('data/top_beers.csv') #dataframe containing top 1000 beers
    ratings_df = pd.read_csv('data/top_ratings.csv') #dataframe containing reviews for top beers
    ratings_df = ratings_df[~(ratings_df.score>5)]  #filter outlier
    ratings_df['uid'] = ratings_df.groupby('username').ngroup() #create user ids for users
    user_df = ratings_df.pivot(index='uid',columns='beer_id',values='score')

    return beer_df, ratings_df, user_df

def add_user_prof(beer_df,user_df,beers,ratings):
    '''
    Takes inputs from user and creates a dictionary containing favorite beers.
    
    Inputs:
        df - user dataframe
        beers - list of beers_ids chosen by user
        
    Returns:
        updated_df - dataframe containing new user row
        uid - user id
    '''
    
    d={}
    for i, b in enumerate(beers):
        id_ = beer_df[beer_df.name==b].iloc[0].beer_id
        d[id_]=ratings[i]
    uid = user_df.shape[0]
    updated_df = user_df.append(pd.DataFrame(d,index=[uid]))
    return updated_df,uid

def similar_users(uid, df, n=5,thresh=1):
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
    # print(user)
    ratings = (~user.isna())
    users_filt = df.T[ratings.values]
    users_filt = users_filt.T.dropna(thresh=thresh)
    # print(f'Users_filt: {users_filt}')
    #create pearson corr matrix
    U = np.corrcoef(users_filt)
    user_idx = len(U)-1
    #find most similar users
    sim_users = np.argsort(U[user_idx][~np.isnan(U[user_idx])])[-(n+1):-1]

    return sim_users

# @st.cache
# def get_user_input(single=True)

if __name__ == '__main__':
    #import dataframes
    beer_df, ratings_df, user_df = load_dataframes()
    breweries = np.sort(beer_df.brewery_name.unique())
    st.title('Welcome to BeerBro!')
    # image = Image.open('img/beer-1796698.jpg')
    # st.image(image)
    # st.write('photo by Helena Lopes') 

    with st.sidebar.title('Find beers similar to your favorites.'):
        brewery = st.sidebar.selectbox(
            'First, pick a brewery:',
            breweries)
        
        beer = st.sidebar.selectbox(
            'Now, pick your favorite beer:',
            beer_df[beer_df.brewery_name==brewery].name)

    # st.sidebar.text_input('Enter a beer you like')
    #beer = input("Enter a beer you like: ")
    # beer = 'Odell IPA'
    #brewery = input("What brewery is it from?: ")
    # brewery = 'Odell'
    
    ####Content Recommender####
    content_r = ContentRecommender(beer_df)
    # content_r.fit()
    beer_recs = content_r.get_recommendations(item=beer,brewery=brewery)
    st.write('Based on the beer you select you may also like:')
    st.table(beer_recs)

    st.sidebar.title('Find beers based on users like you.')
    st.sidebar.write('Give ratings for beers you like:')
    
    beers_=[]
    ratings_=[]
    with st.sidebar.form(key='beer_ratings'):
        # brew1 = st.selectbox('Brewery:', breweries)
        # beers1 = beer_df[beer_df.brewery_name==brew1].name
        beers_.append(st.selectbox('Beer', beer_df.name.sort_values(),key='b1'))
        ratings_.append(st.slider('Rating:',0,5,key='s1'))
        beers_.append(st.selectbox('Beer', beer_df.name.sort_values(),key='b2'))
        ratings_.append(st.slider('Rating:',0,5,key='s2'))
        beers_.append(st.selectbox('Beer', beer_df.name.sort_values(),key='b3'))
        ratings_.append(st.slider('Rating:',0,5,key='s3'))
        beers_.append(st.selectbox('Beer', beer_df.name.sort_values(),key='b4'))
        ratings_.append(st.slider('Rating:',0,5,key='s4'))
        submit_button = st.form_submit_button(label='Get Recommendations')
    
    if submit_button:
        print('creating recommendations...')
        new_df, uid = add_user_prof(beer_df,user_df,beers_,ratings_)
        sim_users = similar_users(uid, new_df,thresh=1)
        
        ####Collaborative Filter Recommender###
        colab_r = CollabRecommender(ratings_df)
        # colab_r.fit(model=SVD, n_factors=100, n_epochs=5, lr_all=0.01, reg_all=0.02)
        # colab_r.predict()
        top_ratings = colab_r.get_top_n()
        user_recs = colab_r.get_recommendations(beer_df,top_ratings,sim_users)
        st.write('Based on the beers you select, other similar users like:')
        st.table(user_recs)
        