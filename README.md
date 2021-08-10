# BeerBro: a beer recommender that knows what beer you will like.
![](img/beer-1796698.jpg)


# Background
I like beer. Who doesn't? There's been countless times I've found myself in the beer isle of a store, wanting to try something new with no idea where to start. I know what beers I've tried, but I can only window shop and usually end up picking a beer based on how cool the label looks. But what if there was some kind of beer recommender I could use that would pick new beers for me?

The goal of this project is just that, build a beer recommender app so I will never again look like a lost and confused shopper in the beer isle.

Introducing the fruits of my labor: **BeerBro**. The BeerBro app combines a content-based natural language processing model with a collaborative matrix factorization model to produce both precise and seridipitous beer recommendations based on a user's preferences.

# Data
To train my model, I used review data from [ratebeer.com](https://www.ratebeer.com/). The dataset was obtained from [Kaggle](https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews?select=beers.csv), and contains more than 9 million reviews on more than 250,000 unique beers.

Each review contains scores for the look,smell,taste, feel, and overall score of each beer. Additionaly, many reviews include a text field which may contain tasting notes or an overall impression of the beer. Also included in the data are descriptions of each beer with features including: style, availability, abv, a text description of the label, the brewery that produced the beer, and its state and/or country.

## Beer Data
Taking a look at the beer data, there are more than 112 unique styles of beer in the dataset. To see their distribution, I plotted the top 10 beer styles by count.

![](img/top_beer_styles.png)
As you might expect, there are a lot of IPAs. (almost 45,000, making up 17% of the total beers)



What states make the most beers?

![](/img/beer_by_state.png)




What's the distribution of alcohol by volume?

![](/img/abv_violin.png)


## Review Data
If I want to build a collaborative filter, I need to look a the reviews themselves. My first concern is if my review-item matrix is dense enough to use a model-based approach for generating reccomendations. As I expected, given that I have more than 9 million reviews and only about 165,000 users and 300,000 beers, my matrix was very sparse with a density less than 1/200%.

```
Number of unique users: 164936
Number of unique beers: 309542
Number of ratings: 9073128

The density is: 0.00018
``` 

To solve this issue, I decided to filter my data by only selecting the top 1000 beers with the most reviews. This yielded a density of 2.5%, which although not ideal, I could work with.
```
Number of unique users: 122712
Number of unique beers: 1000
Number of ratings: 3078488
The density is: 0.02509
```

After filtering my data, I took a look at the distribution of reviews to discover most beers are rated highly with a mean score of 4.01 and std 1.32.

![](img/beer_scores.png)

By filtering the data, I also was able to collect abundant text reviews for each of the 1000 beers. This would prove to helpful for my content-based model.

# Recommender System

![](img/rec_system.png)

## Content Recommender

The content recommender relies on NLP to featurize the text content of the user reviews to find the most similar beers to a user's input. I created a matrix of TF-IDF features using Sklearn's text vectorizer, which I used to compute the cosine similarties between each beer. For each document, I selected to 10,000 highest weighted unigram and bigram words for each document. The resulting item-item similarity matrix is saved to produce future recommendations on the fly.

TF-IDF is a measure of how original a word in a document is by comparing the number of times that word occurs in the doccument with the number of documents having that word. It gives a weight to each word not only by its frequency but its frequency in comparison to all documents, which makes words unique to a document have higher weights than others.

I considered including additional features such as ABV, style, and brewery location with higher weight, however after looking at the text descriptions in addition to the recommendations, the text features appeared to capture these features. The recommendations it produced appeared to be both similar in ABV and style.

## Collaborative Recommender

Because I was dealing with a large (more than 100,000 users and 1000 beers) and sparse (less than 3% density) user-item matrix, I decided to use matrix-factorization to decompose the sparse matrix into user and item utility matrices containing latent features.

<img src="img/matrix_fact.png" alt="Matrix Factorization" width="400"/>
Matrix factorization works by decomposing the user-item matrix into two lower dimensional matrices.

# BeerBro App

![](img/app.png)

