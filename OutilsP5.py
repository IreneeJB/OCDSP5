import pandas as pd
import numpy as np
import missingno as msno
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram

##############################################################################
#  initializePandas() :
#         Aucun paramètres
#
#         Initialise les options pandas
# 
#         Return : None
##############################################################################

def initializePandas() :
    pd.set_option('display.max_columns', 10)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000
    pd.set_option('display.max_colwidth', 30)  # or 199
    return None
    
    
##############################################################################
#  compareColumns(df, L) :
#         df : pd.dataframe
#         L : liste de string de noms de colomnes de data
#
#         Affiche le nombre de valeurs présente dans une colonnes et absentes dans l'autre
#          
#
#         Return : None
##############################################################################

def compareColumns(df, L) :
    for e1 in L :
        for e2 in L:
            if e1 != e2 :
                try :
                    mask = df[e1].notna()
                    print(f'il y a {df[mask][e2].isna().sum()} valeurs dans {e1} qui sont manquantes dans {e2}.')
                except KeyError :
                    print(f"Erreur de clé, couple {e1} - {e2} non traité.")
            else :
                pass
    return None

##############################################################################
#  missingValuesInfos(df) :
#         df : pd.dataframe
#
#         Affiche le nombre de valeurs manquantes, totales, le taux de remplissage et la shape du dataframe
#         Affiche la msno.matrix du dataframe          
#
#         Return : None
##############################################################################

def missingValuesInfos(df) :
    nbRows, nbCols = df.shape
    print(f"Il y a {df.isna().sum().sum()} valeurs manquantes sur {nbRows * nbCols} valeurs totales.")
    print(f"Le taux de remplissage est de : {int(((nbRows*nbCols - df.isna().sum().sum())/(nbRows*nbCols))*10000)/100} %")
    print("Dimension du dataframe :",df.shape)
    msno.matrix(df)
    return None

##############################################################################
# timeCut(df, endDate = None, beginDate = None) :
#         df : dataframe
#         
#         adapté au dataframe "orders_view"
#         Coupe le dataframe orders_view et renvoie les data pour les algos de clustering
#         
#
#         Return : (df_customers, customers_list)
#                  df_customers : renvoie le df
#                  customers_list
##############################################################################

def timeCut(df, end_date = None, start_date = None) :
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], format = "%Y-%m-%d %H:%M:%S")
    
    # Attribution des valeurs par défault
    if end_date == None :
        end_date = df["purchase_date"].max()
    if start_date == None :
        start_date = df['purchase_date'].min()
    today = df["purchase_date"].max()
    
    # Sélection des data concernées
    end_mask = df['purchase_date'] <= end_date
    start_mask = df['purchase_date'] >= start_date
    customers_list = df.loc[end_mask * start_mask, 'customer_unique_id'].unique()
    selected_df = df.loc[end_mask,:]
    
    # Regroupement par client
    df_customers = selected_df.groupby('customer_unique_id').agg({'customer_id' : 'count',
                                                                  'customer_city' : pd.Series.mode,
                                                                  'customer_zip_code_prefix' : 'first',
                                                                  'number_of_items' : ['mean','sum'],
                                                                  'purchase_date' : ['min','max'],
                                                                  'order_price' : ['mean','max','sum'],
                                                                  'total_weight_g' : 'mean',
                                                                  'review_score' : 'mean'})
    # renommage des columns
    df_customers.columns = ["number_of_orders","customer_city",'zip_code_prefix',"items_per_order","number_of_items","first_order_date",
                            "last_order_date","mean_order_price","max_order_price","total_money_spend","mean_weight_order",'satisfaction']
    df_customers["satisfaction"] = df_customers["satisfaction"].round(1)
    
    # Montage du DataFrame pr les modèles
    df_customers['last_order_date'] = pd.to_datetime(df_customers['last_order_date'], format = "%Y-%m-%d %H:%M:%S")
    recency = (today - df_customers["last_order_date"])
    recency = recency.dt.days
    df_data = pd.DataFrame({'recency' : recency,
                            'frequency' : df_customers["number_of_orders"],
                            'monetary' : df_customers["total_money_spend"].round(0).astype(int),
                            'satisfaction' : df_customers["satisfaction"]})
    df_data.sort_index(inplace = True)

    return df_data, customers_list

##############################################################################
#  plot_dendrogram(model, **kwargs) :
#         model : aglomerative clustering
#
#         affiche le dendrogram du modele          
#
#         Return : None
##############################################################################



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)