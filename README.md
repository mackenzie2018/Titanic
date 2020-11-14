# Titanic Competition

After avoiding it for a while, I thought it time to really dive into Kaggle competitions.

The titanic problem as a first port of call seemed logical. As good as any other, at least.

Some general remarks:
- This doc is going to be intentionally unpolished. The aim here is to accurately document my thought process, approach, and hopefully what I've learned as I work through the problem.
- To that end, with the hope of looking back on this one day and laughing, I solemnly swear not to editorialise even the most egregious and humiliating of errors should they arise. Does rubber not screech when it hits the road?

## Run 1: A (Very) Rough Logistic Regression

Being generally pre-disposed to pick the low-hanging fruit, I thought I would start off with a rough and dirty logistic regression. No frills, no messing with the default parameters.

Processing on the `train` and `test` data:
- De-mean and scale the `Age`, `Fare`, `Parch`, and `SibSp` columns.
	- Get these numerical columns on the same scale with zero(ish) mean while preserving the overall distribution of the data.
- Map the `Sex` and `Embarked` columns from `str` to `int`:
	- m/f : 1/0 respectively
	- Q/S/C: 1/2/3 respectively
- Drop `Name`, `Ticket`, and `Cabin`
	- At first glance, the Ticket and Cabin data is fairly messy. Given that `Pclass` already captures the class of ticket it seemed reasonable to drop them for now and focus on the other columns.
	- `Cabin` does tell us where on the ship each passengers lodging was relative to, say, the lifeboats. I'll come back to this in another run.
- Drop the null values.

That's really it. I applied the same transformations to `train` and `test`, and fit the training data to a bare-bones default instance of a `LogisticRegression` classifier from `sklearn.linear_model`.

Predicting and submitting the predictions to Kaggle the result was...

0.76315

Not great, but also not terrible considering.

## Run 2: How about a Support Vector Machine?

Low-hanging fruit remain. Perhaps a large-margin classifier will yield better results. Not knowing the true survival rate aboard the ship, bare intuition tell's us (or tells me, at least) that surviving the Titanic is a low-probability event.

It also tells me that the true classification boundary is more likely to be non-linear than linear.

It may therefore behoove me to try a classifier which is: 
- a bit more cautious and robust to new data than a pure log-reg; and
- is capable of handling non-linear hypotheses.

Choosing the regularisation parameter `C` is a bit arbitrary. Too big and the classifier will be sensitive to outliers in the data (like the few rich passengers paying lots for tickets?).

So I'll start by trying several. Using `C = [0.01, 0.1, 1, 10, 100, 1000]` I got the following survival rate predictions on the `test` set (`random_state=1337` using a radial basis function and default gamma):

C value | predicted survival rate
------- | -----------------------
0.01 	| 0%
0.1  	| 36.3%
1 		| 35.8%
10 		| 27.7%
100 	| 27.3%
1000 	| 30.8%

The first, 0% survival, seems a bit harsh. I also don't need to go to the trouble of teaching a computer to do predict that nobody survived. Then again, it may be useful to know how correct this bleak, bleak prediction is so I'll submit it anyway.

`C` values of 0.1 and 1.0 predict roughly the same survival rate, as do `C` values of 10 and 100. So I'll take one from each.

A `C` value of 1000 is between the two, so I'll also submit these predictions.

Submitting to Kaggle, they scored... 

C value	| predicted survival rate 	| score
-------	| -----------------------	| -----
0.01 	| 0%	| 62.2%
1 		| 35.8%	| 77.0%
10 		| 27.7%	| 77.99%
1000 	| 30.8%	| 77.8%

Interesting.

A 1% improvement on the very basic logistic regression of the same lightly processed `X_train, y_train` data.

We can do a lot better. Back to the drawing board.

## Run 3: A (Slightly) More Nuanced Approach

Having firmly established that mediocre efforts yield barely-better-than-mediocre results it is now time to put more effort into the data processing.

The first two Run's stubborness to move out of the 75% scoring range reflects the crudeness of the approaches taken, particularly in respect of the data itself.

So as to avoid changing too much at once (after all, isn't the point here to learn?) I will stick to incremental changes for now. To wit, I have modestly expanded the feature engineering side of things as follows:
- Replaced the `Embarked` column with three dummy columns.
- Ditto `Pclass`.
- Added `len_name`, the number of characters in the `Name` column. (Hunch: richer people tend to have longer names, richer people more likely to survive.)
	
Starting off with a `LogisticRegression` classifier again using `C = [0.01, 0.025, 0.05, 0.075, 0.1, 1, 10, 100, 1000], random_state=117` and following it up with an `SVM` with the same parameters yielded the following predictions and scores:

C	| LogReg Predicted Survival Rate	| LogReg Score | SVM Predicted Survival Rate | SVM Score
----| -----------------------	| ----- | -------- | ----------------
0.01| 19%	| 77.5% | 0% | not submitted
0.025| 25.1% | 78.7% | 19.1% | 77.5%
0.1 | 34.7%	| 78.2% | 26.6% | 77.7%
1	| 37.6%	| 76.8% | 26.6% | 77.7%
10	| 37.8%	| 77.0% | 29.2% | 78.0%
100	| 37.8%	| not submitted | 34.2% | not submitted
1000| 37.8%	| not submitted | 40.4% | not submitted

We're not flying yet but our eyes have just about managed to peek out over the canopy.

I think it's time to venture out and try other classification algorithms like `DecisionTree`, `KNN`, or `RandomForest`.

## Run4: Vox Populi, Vox Informatio | KNN

Overture: the format of the competition isn't all that great as far as optimisation goes. Not having access to precision/recall/F1-score figures leaves you to follow your nose. Fine, but perhaps not all that scientific? Are they called Data Scientists because Data Janitors wouldn't have anyone clamouring for a career change? 

We press on. First executive decision: sticking with the `train` and `test` data processing from Run 3. 

Now let's take a more democratic approach by letting the people decide via `KNN`. Base intuition: passengers who are similar along several dimensions will be similarly likely to survive or perish.

Initialising the `KNeighboursClassifier` instance with:
- `n_neighbours = [3, 5, 9, 13, 17]`, we'll try a range from 5 to 17. Keep it to odd numbers to avoid tie-breaks.
- `weights = 'distance'`, as we want closer neighbours to have more influence on our predictions
- `algorithm = 'auto'`, let the package take care of this (don't want to over-engineer out of the gate)
- `p=2`, good old Euclidean distance

The results:

neighbours	| predicted survival rate 	| score
-------	| -----------------------	| -----
3 	| 34.9%	| 77.0%
5 	| 35.1%	| 77.3%
9 	| 34.4%	| 77.5%
13 	| 33.4%	| 77.5%
17 | 32.3%	| 77.7%

Perhaps weighting by `distance` wasn't too smart. After all, don't we want to all k neighbours to have equal say?

Running the classifier again with `weights='uniform'`, the results were:

neighbours	| predicted survival rate 	| score
-------	| -----------------------	| -----
3 	| 36.6%	| not submitted
5 	| 34.0%	| 76.1%
9 	| 34.2%	| 76.8%
13 	| 33.0%	| 79.0%
17 | 31.6%	| 78.5%