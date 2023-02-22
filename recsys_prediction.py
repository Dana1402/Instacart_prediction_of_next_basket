import pandas as pd
import multiprocessing
import pickle
from pandas import DataFrame
import datetime as dt
from joblib import Parallel, delayed
import lightgbm as lgb
from lightgbm import LGBMClassifier
import pathlib
from pathlib import Path 

# def load_model(path_to_model):
#     """функция загружает предобученный алгоритм"""
#     with open(path_to_model, 'rb') as file:
#         model = pickle.load(file)
#     return model

class Recsys:
#     num_recs: int
#     df_re: DataFrame

    def __init__(self, num_recs = 10          
#                   path = "/client_part_recsys/"
                ):
        self.path = pathlib.Path.cwd()
        self.num_recs = num_recs
        self.df_products = pd.read_csv(Path(self.path,'products.csv'))
        self.df_transactions = pd.read_csv(Path(self.path,'transactions.csv'))
        self.df_re = self.df_transactions[['user_id', 'order_id', 'order_number', 'product_id']]
#         self.model = load_model()
        self.cpu_count = multiprocessing.cpu_count()
        # загрузка предобученного датасета
        self.prepared_data = pd.read_csv(Path(self.path,'prepared_data.csv'))
        
    def load_model(self):
        """функция загружает предобученный алгоритм"""
        with open(Path(self.path,'lg.pkl'), 'rb') as file:
            model = pickle.load(file)
        return model    

    def build_weiht(self, usr_id=None):
        """функция построения весов для всех продуктов для конкретного пользователя.
        Последние заказы и продукты в них имеют больший вес как более актуальные для пользователя"""
        if usr_id is None:
            user_df = self.df_re
        else:
            user_df = self.df_re[self.df_re.user_id == usr_id]
            
        max_order_prod_usr = user_df[['user_id', 'order_number']].groupby('user_id').agg('max').astype('int8')
        max_order_prod_usr = max_order_prod_usr.rename(columns={'order_number': 'max_order'})
        user_df = user_df.merge(max_order_prod_usr, how='left', on='user_id').fillna(0)
        del max_order_prod_usr
        user_df['weight'] = user_df.order_number / user_df.max_order
        user_df['weight_cum'] = user_df.groupby(['user_id', 'product_id'])['weight'].cumsum()
        user_df.drop(columns=(['max_order', 'weight']), axis=1, inplace=True)        
        user_df.to_csv('prepared_data.csv', index = False)        
        return user_df

    def get_preds(self, usr_id, mode='working'):
        """
        По идентификатору пользователя выдавать набор из K наиболее релевантных для него товаров
        1. Строим веса
        2. добавляем импровизированную колонку с номером последнего заказа
        3. Группируем для получения максимального заказа по продукту
        4. Порядок колонок
        """
        if usr_id in set(self.df_transactions.user_id.values):
            
            if mode == 'demonstration':
                user_df = self.prepared_data[self.prepared_data['user_id'] == usr_id]           
            else:            
                user_df = self.build_weiht(usr_id)

#             max_order = self.df_re[self.df_re.user_id == usr_id]['order_number'].max()
            max_order = user_df[user_df.user_id == usr_id]['order_number'].max()
            user_df = user_df.groupby('product_id').agg('max').reset_index()
            user_df.drop(columns=['order_id', 'order_number'], axis=1, inplace=True)

            user_df['order_number'] = max_order + 1
            cols = ['user_id', 'order_number', 'product_id', 'weight_cum']
            user_df = user_df[cols]

            model = self.load_model()
            user_df['prob'] = model.predict_proba(user_df)[:, 1]
            recs = user_df.sort_values(by='prob', ascending=False)['product_id'].iloc[:self.num_recs]
            recs = list(recs)
            print('finished sleeping @ ', dt.datetime.now())
        else:
            return []
        return recs

    def parallel_counting(self, users, mode='working'):
        """
        Функция для распараллеливания вычислений
        1. Получает количество ядер
        2. Cоздает параллельный экземпляр с указанными ядрами
        3. Вызывает вычисление рекомендаций для списка пользователей
        """
        res = Parallel(n_jobs=self.cpu_count)(delayed(self.get_preds)(i, mode) for i in users)
        return res

    def update_transaction_dataset(self, new_transactions_path):
        """
        Добавить свежие данные о транзакциях.
        Вход - файл с новыми данными в формате csv.
        Сохранить обратно в файл csv
        """
        status = 0
        new_transactions = pd.read_csv(new_transactions_path)

        for i in range(len(new_transactions)):
            if new_transactions.order_id.iloc[i] in self.df_transactions.order_id.values:
                self.df_transactions[self.df_transactions.order_id == new_transactions.order_id.iloc[i]] = \
                new_transactions.iloc[i]
            else:
                self.df_transactions = self.df_transactions.append(new_transactions.iloc[i])
                status += 1
        self.df_transactions.to_csv(Path(self.path , 'transactions_.csv'), index=False)
        if status > 0:
            print('Файл обновлен, добавлено {status} транзакций'.format(status=status))
            print('finished sleeping @ ', dt.datetime.now()) 
        return 'success'

    def update_product_dataset(self, new_products_path):
        """
        Обновить/добавить данные о характеристиках товаров.
        Сохранить обратно в файл csv
        """
        status = 0
        new_products = pd.read_csv(new_products_path)
        for i in range(len(new_products)):
            if new_products.product_id.iloc[i] in self.df_products.product_id.values:
                self.df_products[self.df_products.product_id == new_products.product_id.iloc[i]] = new_products.iloc[i]
            else:
                self.df_products = self.df_products.append(new_products.iloc[i])
                status += 1
        self.df_products.to_csv(Path(self.path ,'products_.csv'), index=False)
        if status > 0:
            print('finished sleeping @ ', dt.datetime.now())
            print('Файл обновлен, добавлено {status} продуктов'.format(status=status))  
        return 'success'

    def build_train_data(self):
        """
        Функция для построения датасета для обучения модели по-новой.
        1. Построение датасетов "корзины клиентов(последний заказ)" и "Предпоследний заказ"
        2. Строим веса
        3. Берем все необходимые данные по id "Предпоследний заказ"
        4. Берем последний заказ по каждому продукту
        5. Проставляем метки, есть ли продукт в последнем заказе
        6. Удаляем ненужные столбцы
        7. Делим данные на обучающий датасет и таргет-метки
        """
        client_basket_ = \
            self.df_transactions.groupby('user_id').apply(lambda x: x[x.order_number == x.order_number.max()])[
                ['user_id', 'product_id']].reset_index(drop=True)
        client_previous_ = \
            self.df_transactions.groupby('user_id').apply(lambda x: x[x.order_number == x.order_number.max() - 1])[
                'order_id']
        cl_b = self.build_weiht()
        cl_b = cl_b[cl_b.order_id.isin(client_previous_)]
        cl_b = cl_b.groupby(['user_id', 'order_id', 'order_number', 'product_id']).agg('max').reset_index()
        cl_b = cl_b.merge(client_basket_.groupby('user_id').agg(list).rename(columns={'product_id': 'basket'}),
                          on='user_id')
        cl_b['is_in_cur_basket'] = 0

        for i in range(len(cl_b)):
            if cl_b['product_id'].iloc[i] in (cl_b['basket'].iloc[i]):
                cl_b['is_in_cur_basket'].iloc[i] = 1
        cl_b.drop(['order_id', 'basket'], axis=1, inplace=True)
        cl_b.to_csv('prepared_data.csv', index = False)          
        X_train, y_train = cl_b.loc[:, cl_b.columns != 'is_in_cur_basket'], cl_b['is_in_cur_basket']

        return X_train, y_train

    def train_model(self):
        """
        Обучить рекомендательную систему заново и сохранить в файл
        """
        X_train, y_train = self.build_train_data()
        lg = LGBMClassifier(num_leaves = 49,
                     max_depth = 2,
                     learning_rate = 0.1,
                     n_estimators = 59,
                     min_child_samples = 20,
                     min_data_in_leaf = 52,
                     class_weight = {0:0.7, 1:0.3}
                   )   
        lg.fit(X_train, y_train)
#         self.model.fit(X_train, y_train)
        pkl_filename = 'lg.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)
        print("Training finished")
        return 'success'





