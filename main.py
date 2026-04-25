#Diana Mtz
#Hana 
#Juan
#Erick Morales

from sklearn.cluster import k_means
import pandas as pd 
import customtkinter as ctk


convs = pd.read_parquet("dataset_50k_anonymized.parquet")

# Reconstruct a conversation
conv = convs[convs["conv_id"] == "some-conv-id"].sort_values("date")

# All conversations for a user
user_convs =convs[convs["user_id"] == "USR-00042"].groupby("conv_id")

print(user_convs.head())



clientes= pd.read_csv('hey_clientes.csv')
productos= pd.read_csv('hey_productos.csv')
transacciones= pd.read_csv('hey_transacciones.csv')