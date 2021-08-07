#general imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import pickle

#content based imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#collaborative imports
from surprise import Dataset
from surprise import Reader
from surprise import dump
from surprise import accuracy
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import KFold
from surprise.model_selection.split import train_test_split




class ContentRecommender():
    '''
    Content Based Beer Recommender
    '''
    def __init__(self, df, measure = cosine_similarity):
        self.sim_d = None
        self.sim_measure = measure
        self.df = df
        # self.rec_d = self.load_obj('similarities')
        try:
            self.rec_d = self.load_obj('similarities')
        except:
            print('If this is your first time running, please use the .fit() method.')

    def fit(self, max_features = 10000):
        '''
        Takes a pandas dataframe containing rating text and creates recommendation dictionary from similarty matrix.
        Pickles recommendation dictionary for future use.
        '''
        tf = TfidfVectorizer(analyzer='word', ngram_range = (1,2),max_features=max_features, stop_words='english')
        print('Fitting...')
        X = tf.fit_transform(self.df.text)
        print('Calculating similarity matrix...')
        cos_sims = self.sim_measure(X,X)
        
        #create similarity dictionary
        rec_d={}
        for idx, row in self.df.iterrows():
            sim_indices = cos_sims[idx].argsort()[-12:-1] #stores 10 most similar
            sim_items = [(cos_sims[idx][i], self.df['beer_id'][i]) for i in sim_indices]
            rec_d[row['beer_id']] = sim_items[1:]
        
        self.save_obj(rec_d,'similarities')
        self.rec_d = rec_d
        
    
    def get_recommendations(self,item,brewery,n=5):
        '''
        Returns df with top n beers related to passed item
        '''
        b_id = self.get_beer_id(item,brewery)    #get id of beer from search
        if b_id:
            sim_beers = self.rec_d[b_id][:n-1:-1]  #return similar beers in descending order
            names=[]
            breweries=[]
            styles=[]
            scores=[]
            abvs=[]
            notes=[]
            states=[]
            countries=[]

            for i, row in enumerate(sim_beers):
                beer = self.df[self.df['beer_id']==row[1]]
                names.append(beer.name.iloc[0])
                breweries.append(beer.brewery_name.iloc[0])
                styles.append(beer['style'].iloc[0])
                scores.append(beer['score'].iloc[0])
                abvs.append(beer['abv'].iloc[0])
                notes.append(beer['notes'].iloc[0])
                states.append(beer['state'].iloc[0])
                countries.append(beer['country'].iloc[0])
                # print(f'Reccomendation #{i+1}: {beer_name} from {brewery}')
            d = {'name':names,'brewery':breweries,'style':styles,'rating':scores,
                'abv':abvs, 'description':notes,'state':states,'country':countries}
            new_df = pd.DataFrame(d)
            print(new_df)
            return new_df
    
    def get_beer_id(self,item,brewery):
        '''
        Given beer name, returns beer id if a match is found
        '''
        beer = self.df[((self.df['name'].str.contains(item))&(self.df['brewery_name'].str.contains(brewery)))]

        if len(beer)<1:
            print('Sorry, no beer was found with that name and brewery. \n Please try again.')

        elif len(beer)>1:
            print('More than one beer found for that name and brewery. \n Please be more specific.')
        
        else:
            return beer.beer_id.iloc[0]

        return False
    
    #pickle helper functions
    def save_obj(self,obj, name):
        with open('obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self,name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)         

class CollabRecommender():
    '''
    SVD Collaborative-Based Beer Recommender
    '''
    def __init__(self, df):
        reader = Reader(rating_scale=(1,5))
        self.data = Dataset.load_from_df(df[['username','beer_id','score']],reader)
        # self.trainset = data.build_full_trainset()

    def fit(self, model=SVD, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        trainset = self.data.build_full_trainset()
        model = model(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, verbose=True)
        model.fit(self.trainset)
    
    def build_user_profile(self):
        pass




        



if __name__ == '__main__':
    
    #import dataframes
    print('Importing top beers...')
    # beer_df = pd.read_csv('data/top_beers.csv') #dataframe containing top 1000 beers
    ratings_df = pd.read_csv('data/top_ratings.csv') #dataframe containing reviews for top beers
    ratings_df = ratings_df[~(ratings_df.score>5)]
    # beer = input("Enter a beer you like: ")
    # brewery = input("What brewery is it from?: ")
    # content_r = ContentRecommender(beer_df)
    # content_r.fit()
    # content_r.get_recommendations(item=beer,brewery=brewery)

    colab_r = CollabRecommender(ratings_df)




    
    