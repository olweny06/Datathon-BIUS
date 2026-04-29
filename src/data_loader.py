import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')
dataset_path = "D:\\DATATHON\\Datathon-BIUS\\dataset"

def load_data():
    products = pd.read_csv(f"{dataset_path}\\products.csv")
    customers = pd.read_csv(f"{dataset_path}\\customers.csv", parse_dates =['signup_date'])
    geography = pd.read_csv(f"{dataset_path}\\geography.csv")
    orders = pd.read_csv(f"{dataset_path}\\orders.csv", parse_dates = ['order_date'])
    order_items = pd.read_csv(f"{dataset_path}\\order_items.csv")
    payments = pd.read_csv(f"{dataset_path}\\payments.csv")
    promotions = pd.read_csv(f"{dataset_path}\\promotions.csv", parse_dates = ["start_date", "end_date"])
    shipments = pd.read_csv(f"{dataset_path}\\shipments.csv", parse_dates = ['ship_date','delivery_date'])
    returns = pd.read_csv(f"{dataset_path}\\returns.csv", parse_dates = ['return_date'])
    reviews = pd.read_csv(f"{dataset_path}\\reviews.csv", parse_dates = ['review_date'])
    sales = pd.read_csv(f"{dataset_path}\\sales.csv", parse_dates = ['Date'])
    web_traffic = pd.read_csv(f"{dataset_path}\\web_traffic.csv", parse_dates = ['date'])
    inventory = pd.read_csv(f"{dataset_path}\\inventory.csv", parse_dates = ['snapshot_date'])
    
    return products, customers, geography, orders, order_items, payments, promotions, shipments, returns, reviews, sales, web_traffic, inventory


    