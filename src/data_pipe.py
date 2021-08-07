import pandas as pd
import numpy as np
#pyspark imports
import pyspark as ps
from pyspark.sql.types import *
from pyspark.sql.functions import countDistinct


def merge_data(beer_path,brewery_path,ratings_path,filename):
    '''
    Cleans and merges beer, brewery and top rating text into a single csv

    Inputs:
        beer_path - path to beers.csv
        brewery_path - path to breweries.csv
        ratings_path - path to top ratings csv
        filename - name of output file
    Returns:    None
    '''
    print('Reading beers...')
    beer_df = pd.read_csv(beer_path)
    print('Reading breweries...')
    brewery_df = pd.read_csv(brewery_path)
    print('Reading reviews...')
    reviews_df = pd.read_csv(ratings_path)

    #drop unwanted columns
    beer_df.drop(['retired','availability'], axis=1, inplace=True)
    brewery_df.drop(['city','state','country','notes','types'], axis=1, inplace=True)
    
    means = reviews_df.groupby('beer_id').mean()  #calculate mean score per beer
    reviews_df = reviews_df[~(reviews_df.text=='\xa0\xa0')] #remove empty text rows
    reviews_df = reviews_df.groupby('beer_id').text.apply(lambda x: ','.join(x)).reset_index() #join text on each beer
    reviews_df = reviews_df.merge(means,on='beer_id') #merge mean score with text
    #merge dataframes into one
    df = reviews_df.merge(beer_df, how = 'left',left_on='beer_id', right_on='id')
    df = df.merge(brewery_df, how = 'left',left_on='brewery_id', right_on='id')
    # print(df.columns[0])
    #drop added columns
    df.drop(['id_x','id_y'],axis=1,inplace=True)
    df.rename({'name_x':'name','name_y':'brewery_name'},axis=1,inplace=True)
    print('Saving to file...')
    df.to_csv(filename, index=False)

def get_top_ratings(path, filename, n=1000):
    '''
    Uses Spark to load in large ratings dataset and filter the top n rated items.
    Saves filtered ratings as csv
    
    Inputs:
        path - path to ratings csv
        filename - path and name of output file
        n - number of most rated reviews to filter
    Returns:    None
    '''
    
    #initialize Spark Session
    spark = ps.sql.SparkSession.builder \
            .master("local[*]") \
            .config("spark.driver.memory", "15g") \
            .appName("BeerBro") \
            .getOrCreate()
    #define schema
    users_schema = StructType( [
    StructField('beer_id',IntegerType(),True),
    StructField('username',StringType(),True),
    StructField('date',DateType(),True),
    StructField('text',StringType(),True),
    StructField('look',FloatType(),True),
    StructField('smell',FloatType(),True),
    StructField('taste',FloatType(),True),
    StructField('feel',FloatType(),True),
    StructField('overall',FloatType(),True),
    StructField('score',FloatType(),True), ] )
    print('Reading reviews.csv...')
    ratings = spark.read.csv('data/reviews.csv',
                            header=True,
                            schema=users_schema,
                            inferSchema=False)
    top_ratings = ratings.groupBy('beer_id').count().orderBy("count",ascending=False).limit(n)
    ratings = ratings.join(top_ratings,on='beer_id')
    ratings = ratings.drop('date','look','smell','taste','feel','overall','count')
    print('Converting to Pandas DataFrame...')
    df = ratings.toPandas()
    spark.stop()

    df.dropna(inplace=True)
    df = df[~(df.score>5)] #drop any ratings above 5
    df.to_csv(filename,index=False)



if __name__ == '__main__':
    
    #define paths to data files
    beers_path = 'data/beers.csv'
    brewery_path = 'data/breweries.csv'
    review_path = 'data/reviews.csv'

    # get_top_ratings('data/reviews.csv', 'data/top_ratings.csv')
    merge_data('data/beers.csv', 'data/breweries.csv', 'data/top_ratings.csv', 'data/top_beers.csv')    


