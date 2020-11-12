# Titanic Competition

After avoiding it for a while, I thought it time to really dive into Kaggle competitions.

The titanic problem as a first port of call seemed logical. As good as any other, at least.

## Attempt 1: A (Very) Rough Logistic Regression

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

## Attempt 2: How about a Support Vector Machine?

Low-hanging fruit remain. Perhaps a large-margin classifier will yield better results. Not knowing the true survival rate aboard the ship, bare intuition tell's us (or tells me, at least) that surviving the Titanic is a low-probability event.

It also tells me that the classification boundary is more likely to be non-linear than linear.

It may therefore behoove me to try a classifier which is: 
- a bit more cautious and robust to new data than a pure log-reg; and
- is capable of handling non-linear hypotheses.

Choosing the regularisation parameter `C` is a bit arbitrary. Too big and the classifier will be sensitive to outliers in the data (like the few rich passengers paying lots for tickets?).

So I'll start try several. Using `C = [0.01, 0.1, 1, 10, 100, 1000]` I got the following survival rate predictions on the `test` set (`random_state=1337` using a radial basis function and default gamma):

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