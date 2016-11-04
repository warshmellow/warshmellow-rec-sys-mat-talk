# warshmellow-rec-sys-mat-talk
Notes for Recommendation Systems and Matrix Factorization

## Value of Recommendation Systems vs. Traditional Recommendation: the Long Tail Problem

Suppose you opened a movie store IRL, you have shelf space for 1,000 movies and you're deciding which movies you should put. Each movie is an `item`; each customer is a `user`.

![Movie store IRL](https://flavorwire.files.wordpress.com/2012/04/kims.jpeg)

Several choices:

1. 1000 copies of last year's most popular movie
1. 1000 most popular movies from last 30 years
1. your favorite 1000 movies
1. etc.

In all these cases, and any other case, you're limited to the shelf space of 1,000 movies, and there will be movies that you may know about but the customer will never even think about because you don't have enough space. What do you do about those movies that never make it to the shelves?

![Long Tail](http://www.thelongtail.com/conceptual.jpg)

Let's say we could measure the interest customers have in a movie by a single number. If you take enough movies, you can construct a graph with x-axis being the movie and y-axis being the interest, sorting by most interesting movie to the left, rightward. No matter which combination of movies you pick for your store, you will be limited to the movies more interesting than the least interesting movie in your store.

This graph has a tail, but you cut it off. How do you access the movies to the right of the cutoff? That's the `Long Tail Problem`.  Recommendation systems allow you to access the entire graph to show customers.


![Movie store IRL](https://flavorwire.files.wordpress.com/2012/04/kims.jpeg)

vs.

![Movie store online](http://www.essentialdesigns.net/wp-content/uploads/2015/09/netflixandchill11.gif)


## The Netflix Problem

There are two entities: `users` and `items`. Each `user` can give a positive rating in a small range (such as 1-5) to each `item`, but does not have to, in which case the rating is blank.

We want to guess each blank rating. For each blank rating for each user and item, this will tell us a guess for what the user would rate the item if they didn't yet rate.

### Formulation using Utility Matrix
![Utility Matrix](http://bit.ly/2eHEPyU)

We can form a sparse matrix `M` where the rows are `users` and the columns are `items` and each entry is the rating, and 0 where there is no rating. The matrix is sparse because almost every entry is 0. This matrix is the `Utility Matrix`.

Take a reasonable case of say 1,000,000 users and 20,000 movie where most users will have seen fewer than 500 movies.
### Evaluation by Root Mean Square Error

Your goal is to find another matrix `N` that is not sparse (i.e. that has a guess for every rating by each user for each item) that is "close" to `M`. Closeness is measured by `Root Mean Square Error (RMSE)`. Roughly it measures how far of the guess rating will be from the real thing on average.

Mathematically you take `M - N`, square all the entries, add them up then take the square root.

So what's the Problem? Netflix had its own proprietary recommendation system to compute `N` given `M` called `CineMatch`. You were asked to, given `M` and an unknown test set `M'` which consisted of rows of `M` build a system to compute `N'`, a guess of `M'` whose `RMSE` was at most 90% of what CineMatch would compute. In other words, some ground truth ratings are hidden from your recommendation system and you have to guess them on average better than CineMatch.

## Two Approaches to the Netflix Problem
Generally two approaches are taken for this problem.

1. Content-based recommendation
1. Collaborative-filtering based recommendation

Content-based asks you to roughly speaking categorize items recommend items based on these categories.

Collaborative-filtering asks you to guess ratings of items by users based on similarity of users and their ratings.

## Low Rank Matrix Factorization (UV-factorization)
![Spark and ALS](http://i.picresize.com/images/2016/11/04/62zpb.png)

One class of solutions we'll look at is based on a collaborative-filtering technique called `Low Rank Matrix Factorization`.

The goal is to, given utility matrix `M`, compute two matrices `U`, `V` such that `N = UV`, the RMSE between `M` and `N` is as small as possible, and `U` is "low rank", which means its number of columns `k` is much lower than the number of items. Roughly speaking, we're saying that a user's rating of an item is determined by a (linear) combination of `k` factors. Note that here we have to fix `k` from the beginning, then find `U` and `V`.

## Interpreting UV-factorization
The `k` columns are roughly speaking clusters of movies. Think of them as genres and each item and user partakes in a scoring of a genre. These genres may or may not make sense to humans, like horror movies and rom-coms, vs something the computer just mined for you.

## Computing the UV-factorization by Gradient Descent
![Gradient Descent](http://bit.ly/2eHFn89)

Roughly speaking, we can formulate the calculation of `N` as an optimization problem: find `U` and `V` such that `N = UV` and that minimizes RMSE of `M` and `N`. Because the square of RMSE is twice differentiable (it is the sum of squares) and because minimizing that square is the same as minimizing the RMSE, we can apply techniques of `convex optimization`. One algorithmic way to solve this optimization problem is called `Gradient Descent`. Roughly speaking, you start with guesses of `U` and `V` and then you adjust the entries of `U` and `V` in the "direction" of the greatest decrease of RMSE, i.e., in the direction of the gradient of an expression of RMSE. Eventually you'll hit a possible pair of `U` and `V` where the algorithm will stop. This may or may not be the best pair, so you may have to run the algorithm several times. In optimization speak, you only get local minima.

## Spark and ALS Recommendations
We use `Apache Spark` and it comes with one Recommender System algorithm called `Alternating Least Squares (ALS)`.

Very roughly speaking, the `ALS` algorithm differs from a `Gradient Descent` algorithm in that it alternately fixes `U` and `V` and optimizes the other. For a variety of mathematical reasons, the optimization can be solved by repeatedly using multiplication and inverses of several very low dimensional matrices.

There are several benefits to this, chiefly being

1. easier to parallelize
1. can work with not so sparse matrices where `Gradient Descent` would be infeasible.

## Explicit vs Implicit Feedback
![Explicitly](http://www.guidingtech.com/assets/postimages/2015/08/netflix-rating-reset.png)

vs.

![Implicitly](http://cdn.bgr.com/2016/02/scooby-doo-netflix.jpg?quality=98&strip=all&w=624)

`Explicit feedback` is where the user explicitly gave a rating. Getting this data may actually be infeasible. In the extreme case, if you start with a dataset in which every rating is 0, how would you make recommendations at all? Any situation in which you don't have enough data at the beginning is called the `Cold Start Problem`.

One way to get around this problem is to create recommendation systems off of `Implicit Feedback`. This is where the entries of the Utility Matrix are not ratings, but could be better interpreted as "confidence". This could be as simple as the number of impressions the user has made on a product page for a movie, rather than an explicit rating the user made after watching a movie.

A lot of implicit feedback may be taken for the users, so the utility matrix will not be as sparse as one with explicit feedback. The `ALS` Algorithm is designed expressly for this case.

## Serving Recommendations Fast
Exercise for the reader :P

## References
[Mining Massive Datasets, Chapter 9](http://www.mmds.org/)

[Collaborative Filtering for Implicit Feedback Datasets (pdf)](http://yifanhu.net/PUB/cf.pdf)
