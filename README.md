# Route climbing recommender and analysis

Two data science projects exploring 116k climbing routes from [Mountain Project](https://www.mountainproject.com/). 
📖 Blog posts: [What drives climbing ratings?](https://arianedekker.github.io/blog/MP_Project1) · [Route recommendation](https://arianedekker.github.io/blog/MP_Project2)

---
 
## Project 1 — What drives climbing ratings?
`Climbing_rate_regression.py`
 
Extract climbing style features from route descriptions (e.g. movement style, rock angle and features, hold rypes) using TF-IDF vectorization. Performed regression analysis of community average star rating on grade, popularity, and style → **Pumpy routes are the strongest positive predictor of star rating.**
 
## Project 2 — Personalized Route Recommendation
`Route_recommender.py`
 
A dual-input neural network (LSTM on route descriptions + MLP on structured features) trained in two stages: global pretraining on community ratings, then transfer learning on personal ticks. Outputs rank route and area recommendations filtered to your grade range and preferred style.
 
---
 
## Setup
 
Download the Mountain Project dataset from [Kaggle](https://www.kaggle.com/datasets/pdegner/mountain-project-rotues-and-forums). For the recommender, export your personal ticks from Mountain Project.
