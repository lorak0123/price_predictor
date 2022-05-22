# Price predictor

The script is responsible for prediction of cryptocurrency price values. When the script is running, a DataFrame object is prepared, of which the price is also a set `X`, and a price column is created as a set `y` which is an offset of the column from the set `X` by the specified number of days (by default 30 days). 

Next, the `X_predict` object is created, containing data on the basis of which the prediction will be made (by default, it is the last 30 days from the set `X`, for which the newly created object `y` has zero values resulting from the shift.
The `SVR (Support Vector Regression)` model is then created from the scikit-learn library and learned.

A prediction based on the variable `X_predict` is performed on the learned model.
