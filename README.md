# Instacart_prediction_of_next_basket
This repository contains notebooks which intend to solve the task of the next basket prediction basing on provided data of transactions and products

The target was to cross score 0.25 in the Kaggle competition 
Jupiter notebook contains several binary classification algorithms plus optimisation some of them. 
Training was made on last minus one orders. Target data are the last orders. 

Main.py contains recsys class with methods:
 * get prediction for one user
 * get prediction for several users (used parallel counting)
 * update transactions and products
 * retrain model
