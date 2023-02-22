# Instacart_prediction_of_the_next_basket
This repository contains notebook which intends to solve the task of the next basket prediction basing on provided data of transactions and products

The target was to cross score 0.25 in the Kaggle competition https://www.kaggle.com/competitions/skillbox-recommender-system/submissions
Jupiter notebook contains several binary classification algorithms plus optimisation some of them. 
Training was made on last minus one orders. Target data are the last orders. 
My result is ~0.31

Initially this task can be divided into two components:

1. Determine as accurately as possible which products the customer will reorder. Usually ppl buy the same things each time so they will buy them in the future. Also weight was added to make an accent on the last orders because one could buy products in the past and then with time stop to buy them. So I considered a relevant choice with the bigger weight.
2. Add products that may not be in the user's purchase history but may be purchased next time. I tried several possible variants:
Collaborative filtering(time consuming as we have 100K users). Associative rules(What is ordered with a product that is in the predicted basket.)

Main.py contains recsys class with methods:
 * get prediction for one user
 * get prediction for several users (used parallel counting)
 * update transactions and products
 * retrain model


To try this recsys just launch Run_it.py
Prepared_data.csv for demonstration
