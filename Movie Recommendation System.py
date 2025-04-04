# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:40:35 2023

@author: lEO
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import math
# Import ML Libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')

# Dealing with warnings and importing the dataset
data = pd.read_csv('C:/Users/lEO/Desktop/Dataset/Kaggle Datasets/movies_genre.csv')
dataset = pd.read_csv('C:/Users/lEO/Desktop/Dataset/Kaggle Datasets/movies_genre.csv')

genre = dataset.iloc[:, 4:]

association_support = apriori(genre, min_support = 0.01, use_colnames = True)
association_confidence = association_rules(association_support, metric = "confidence", min_threshold = 0.1)
association_lift = association_rules(association_support, metric = "lift", min_threshold = 1.25)

# Movies Classified based on Genre
movie_western = [{movie: info} for movie, info in zip(dataset.loc[dataset.Western == 1, 'title'], dataset.loc[dataset.Western == 1, 'overview'])]
movie_romance = [{movie: info} for movie, info in zip(dataset.loc[dataset.Romance == 1, 'title'], dataset.loc[dataset.Romance == 1, 'overview'])]
movie_horror = [{movie: info} for movie, info in zip(dataset.loc[dataset.Horror == 1, 'title'], dataset.loc[dataset.Horror == 1, 'overview'])]
movie_tv_movie = [{movie: info} for movie, info in zip(dataset.loc[dataset.loc[:, "TV Movie"] == 1, 'title'], dataset.loc[dataset.loc[:, "TV Movie"] == 1, 'overview'])]
movie_fantasy = [{movie: info} for movie, info in zip(dataset.loc[dataset.Fantasy == 1, 'title'], dataset.loc[dataset.Fantasy == 1, 'overview'])]
movie_drama = [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1, 'title'], dataset.loc[dataset.Drama == 1, 'overview'])]
movie_scifi = [{movie: info} for movie, info in zip(dataset.loc[dataset["Science Fiction"] == 1, 'title'], dataset.loc[dataset["Science Fiction"] == 1, 'overview'])]
movie_war = [{movie: info} for movie, info in zip(dataset.loc[dataset.War == 1, 'title'], dataset.loc[dataset.War == 1, 'overview'])]
movie_music = [{movie: info} for movie, info in zip(dataset.loc[dataset.Music == 1, 'title'], dataset.loc[dataset.Music == 1, 'overview'])]
movie_history = [{movie: info} for movie, info in zip(dataset.loc[dataset.History == 1, 'title'], dataset.loc[dataset.History == 1, 'overview'])]
movie_thriller = [{movie: info} for movie, info in zip(dataset.loc[dataset.Thriller == 1, 'title'], dataset.loc[dataset.Thriller == 1, 'overview'])]
movie_comedy = [{movie: info} for movie, info in zip(dataset.loc[dataset.Comedy == 1, 'title'], dataset.loc[dataset.Comedy == 1, 'overview'])]
movie_crime = [{movie: info} for movie, info in zip(dataset.loc[dataset.Crime == 1, 'title'], dataset.loc[dataset.Crime == 1, 'overview'])]
movie_mystery = [{movie: info} for movie, info in zip(dataset.loc[dataset.Mystery == 1, 'title'], dataset.loc[dataset.Mystery == 1, 'overview'])]
movie_action = [{movie: info} for movie, info in zip(dataset.loc[dataset.Action == 1, 'title'], dataset.loc[dataset.Action == 1, 'overview'])]
movie_animation = [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1, 'title'], dataset.loc[dataset.Animation == 1, 'overview'])]
movie_family = [{movie: info} for movie, info in zip(dataset.loc[dataset.Family == 1, 'title'], dataset.loc[dataset.Family == 1, 'overview'])]
movie_adventure = [{movie: info} for movie, info in zip(dataset.loc[dataset.Adventure == 1, 'title'], dataset.loc[dataset.Adventure == 1, 'overview'])]

# Movies Classified according to a lift greater than 1.25 [GROUPS OF TWO]
lift1_animation_comedy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Comedy == 1].loc[dataset.Animation == 1, "title"], dataset.loc[dataset.Comedy == 1].loc[dataset.Animation == 1, "overview"])]
lift1_animation_adventure =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Adventure == 1].loc[dataset.Animation == 1, "title"], dataset.loc[dataset.Adventure == 1].loc[dataset.Animation == 1, "overview"])]
lift1_adventure_family =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Family == 1].loc[dataset.Adventure == 1, "title"], dataset.loc[dataset.Family == 1].loc[dataset.Adventure == 1, "overview"])]
lift1_family_comedy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Comedy == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Comedy == 1].loc[dataset.Family == 1, "overview"])]
lift1_family_animation =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Comedy == 1].loc[dataset.Family == 1, "overview"])]
lift1_action_thriller =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Thriller == 1].loc[dataset.Action == 1, "title"], dataset.loc[dataset.Thriller == 1].loc[dataset.Action == 1, "overview"])]
lift1_drama_thriller =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Thriller == 1].loc[dataset.Drama == 1, "title"], dataset.loc[dataset.Thriller == 1].loc[dataset.Drama == 1, "overview"])]
lift1_crime_action =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Action == 1].loc[dataset.Crime == 1, "title"], dataset.loc[dataset.Action == 1].loc[dataset.Crime == 1, "overview"])]
lift1_drama_crime =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Crime == 1].loc[dataset.Drama == 1, "title"], dataset.loc[dataset.Crime == 1].loc[dataset.Drama == 1, "overview"])]
lift1_drama_comedy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Comedy == 1].loc[dataset.Drama== 1, "title"], dataset.loc[dataset.Comedy == 1].loc[dataset.Drama == 1, "overview"])]
lift1_crime_thriller =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Thriller == 1].loc[dataset.Crime == 1, "title"], dataset.loc[dataset.Thriller == 1].loc[dataset.Crime == 1, "overview"])]
lift1_crime_action =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Action == 1].loc[dataset.Crime == 1, "title"], dataset.loc[dataset.Action == 1].loc[dataset.Crime == 1, "overview"])]
lift1_family_fantasy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Fantasy == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Fantasy == 1].loc[dataset.Family == 1, "overview"])]
lift1_animation_fantasy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Fantasy == 1].loc[dataset.Animation == 1, "title"], dataset.loc[dataset.Fantasy == 1].loc[dataset.Animation == 1, "overview"])]
lift1_adventure_fantasy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Fantasy == 1].loc[dataset.Adventure == 1, "title"], dataset.loc[dataset.Fantasy == 1].loc[dataset.Adventure == 1, "overview"])]
lift1_mystery_thriller =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Thriller == 1].loc[dataset.Mystery == 1, "title"], dataset.loc[dataset.Thriller == 1].loc[dataset.Mystery == 1, "overview"])]
lift1_action_scifi =  [{movie: info} for movie, info in zip(dataset.loc[dataset['Science Fiction'] == 1].loc[dataset.Action == 1, "title"], dataset.loc[dataset["Science Fiction"] == 1].loc[dataset.Action == 1, "overview"])]
lift1_action_adventure =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Adventure == 1].loc[dataset.Action == 1, "title"], dataset.loc[dataset.Adventure == 1].loc[dataset.Action == 1, "overview"])]
lift1_adventure_scifi =  [{movie: info} for movie, info in zip(dataset.loc[dataset["Science Fiction"] == 1].loc[dataset.Adventure == 1, "title"], dataset.loc[dataset["Science Fiction"] == 1].loc[dataset.Adventure == 1, "overview"])]
lift1_horror_thriller =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Thriller == 1].loc[dataset.Horror == 1, "title"], dataset.loc[dataset.Thriller == 1].loc[dataset.Horror == 1, "overview"])]
lift1_drama_history =  [{movie: info} for movie, info in zip(dataset.loc[dataset.History == 1].loc[dataset.Drama == 1, "title"], dataset.loc[dataset.History == 1].loc[dataset.Drama == 1, "overview"])]
lift1_war_history =  [{movie: info} for movie, info in zip(dataset.loc[dataset.History == 1].loc[dataset.War == 1, "title"], dataset.loc[dataset.History == 1].loc[dataset.War == 1, "overview"])]
lift1_romance_comedy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]
lift1_romance_drama =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Romance == 1, "overview"])]

# Movies Classified according to a lift greater than 1.25 [GROUPS by 3]
lift2_animation_comedy_family =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Comedy == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Animation == 1].loc[dataset.Comedy == 1].loc[dataset.Family == 1, "overview"])]
lift2_animation_adventure_family =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Adventure == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Animation == 1].loc[dataset.Adventure == 1].loc[dataset.Family == 1, "overview"])]
lift2_animation_adventure_fantasy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Adventure == 1].loc[dataset.Fantasy == 1, "title"], dataset.loc[dataset.Animation == 1].loc[dataset.Adventure == 1].loc[dataset.Fantasy == 1, "overview"])]
lift2_action_thriller_crime =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Action == 1].loc[dataset.Thriller == 1].loc[dataset.Crime == 1, "title"], dataset.loc[dataset.Action == 1].loc[dataset.Thriller == 1].loc[dataset.Crime == 1, "overview"])]
lift2_drama_thriller_crime =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Thriller == 1].loc[dataset.Crime == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Thriller == 1].loc[dataset.Crime == 1, "overview"])]
lift2_drama_comedy_romance =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]

lift2_drama_war_history =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.War == 1].loc[dataset.History == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.War == 1].loc[dataset.History == 1, "overview"])]
# lift2_drama_comedy_romance =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]
# lift2_drama_comedy_romance =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]
# lift2_drama_comedy_romance =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]
# lift2_drama_comedy_romance =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]
# lift2_drama_comedy_romance =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "title"], dataset.loc[dataset.Drama == 1].loc[dataset.Comedy == 1].loc[dataset.Romance == 1, "overview"])]

# Movies Classified according to a lift greater than 1.25 [GROUPS by 4]
lift3_animation_comedy_family_adventure =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Comedy == 1].loc[dataset.Adventure == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Animation == 1].loc[dataset.Comedy == 1].loc[dataset.Adventure == 1].loc[dataset.Family == 1, "overview"])]
lift3_animation_comedy_family_fantasy =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Comedy == 1].loc[dataset.Fantasy == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Animation == 1].loc[dataset.Comedy == 1].loc[dataset.Fantasy == 1].loc[dataset.Family == 1, "overview"])]
lift3_animation_fantasy_family_adventure =  [{movie: info} for movie, info in zip(dataset.loc[dataset.Animation == 1].loc[dataset.Fantasy == 1].loc[dataset.Adventure == 1].loc[dataset.Family == 1, "title"], dataset.loc[dataset.Animation == 1].loc[dataset.Fantasy == 1].loc[dataset.Adventure == 1].loc[dataset.Family == 1, "overview"])]

# YOU STOPPED AT 231 at LIFT




