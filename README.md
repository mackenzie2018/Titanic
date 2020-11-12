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
	- `Cabin` does tell us where on the ship each passengers lodging was to, say, the lifeboats. I'll come back to this in another run.
- Drop the null values.

That's really it. I applied the same transformations to `train` and `test`, and fit the training data to a bare-bones default instance of a `LogisticRegression` classifier from `sklearn.linear_model`.

Predicting and submitting the predictions to Kaggle the result was...

0.76315

Not great, but also not terrible considering.