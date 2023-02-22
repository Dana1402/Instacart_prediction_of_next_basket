import json
import requests
import recsys_prediction
from recsys_prediction import Recsys
# import keyboard  # using module keyboard

recsys = Recsys()

# Инструкции
action = input("Enter action:\n\
           * get_prediction_for_user: press 1,\n\
           * get_prediction_for_many_users: press 2,\n\
           * train_model: press 3,\n\
           * add_new_products: press 4,\n\
           * add_new_transactions: press 5\n")

if action == '1':
    user = int(input("Enter user ID: "))
    preds = recsys.get_preds(usr_id = user, mode = 'demonstration')
    print('Next basket for user {} : {}'.format(user, preds))
elif action == '2':
    users = input('Enter user IDs (in format 1,2,3)')
    users = [int(n) for n in users.split(',')]
    preds = recsys.parallel_counting(users, mode = 'demonstration')
    print('Next basket for {} users: {} '.format(users, preds))
elif action == '3':
    recsys.train_model()
elif action == '4':
    path = input('Enter file name: ')
    if len(path) > 0: 
        recsys.update_product_dataset(path)
    else:
        print('Path not found')
elif action == '5':
    path = input('Enter file name: ')
    if len(path)>0: 
        recsys.update_transaction_dataset(path)
    else:
        print('Path not found')
else:
    'Метод отсутствует'






