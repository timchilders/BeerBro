#general imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

#content based imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#collaborative imports
from surprise import Dataset
from surprise import Reader
from surprise import dump
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import KFold
from surprise.model_selection.split import train_test_split




class ContentRecommender():
    '''
    Content Based Beer Recommender
    '''
    def __init__(self, measure = cosine_similarity):
        self.sim_d = None
        self.sim_measure = measure
        self.beer_df = pd.read_csv('data/beer_features.csv')
        try:
            with open('data/simalarites.json','r') as f:
                self.rec_d = f.read()
            f.close()
        except:
            print('If this is your first time running, please use the .fit() method.')

    def fit(self, df, max_features = 10000):
        '''
        Takes a pandas dataframe containing rating text and creates recommendation dictionary from similarty matrix.
        Saves recommendation dictionary as json file for future use.
        '''
        self.save=save
        self.tf = TfidfVectorizer(analyzer='word', ngram_range = (1,2),max_features=max_features, stop_words='english')
        print('Fitting...')
        X = tf.fit_transform(df.text)
        print('Calculating simalrity matrix...')
        cos_sims = self.sim_measure(X,X)
        
        #create similarity dictionary
        rec_d={}
        for idx, row in df.iterrows():
            sim_indices = cosine_sims[idx].argsort()[-12:-1] #stores 10 most similar
            sim_items = [(cosine_sims[idx][i], df['beer_id'][i]) for i in sim_indices]
            rec_d[row['beer_id']] = sim_items[1:]
        
        with open('data/simalarites.json','w') as f:
            json.dump(sim_d, f)
        f.close()
        self.rec_d = rec_d
        

    
    def get_recommendations(self,item,n=5):
        '''
        Returns top n beers related to passed item as dictionary
        '''
        item = get_beer_id(item)
        df = self.beer_df
        sim_beers = results[item][:n-1:-1]
        names=[]
        breweries=[]
        styles=[]
        scores=[]
        abvs=[]
        notes=[]
        states=[]
        countries=[]

        for i, row in enumerate(sim_beers):
            beer = df[df['beer_id']==row[1]]
            names.append(beer.name.iloc[0])
            breweries.append(beer.brewery_id.iloc[0])
            styles.append(beer['style'].iloc[0])
            scores.append(beer['score'].iloc[0])
            abvs.append(beer['abv'].iloc[0])
            notes.append(beer['notes'].iloc[0])
            states.append(beer['state'].iloc[0])
            countries.append(beer['country'].iloc[0])
            print(f'Reccomendation #{i+1}: {beer_name} from {brewery}')
        d = {'name':names,'brewery':breweries,'style':styles,'rating':scores,
            'abv':abvs, 'description':notes,'state':states,'country':countries}
        new_df = pd.DataFrame(d)
        return new_df
    
    def get_beer_id(self,item):
        '''
        Given beer name, returns beer id if a match is found
        '''
        



if __name__ == '__main__':
    
    #import dataframes
    beer_df = pd.read_csv('data/beer_features.csv') #dataframe containing top 1000 beers
    
    
    ratings_df = pd.read_csv('data/top_rated.csv')
    df = ratings_df.dropna()
    df = df[~(df.score>5)]
    df = df.drop(df.columns[0],axis=1)
    df = df.drop('count')
    