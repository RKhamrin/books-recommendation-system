def toppop_model(users_items_ratings, item_col):
    best_seller = users_items_ratings[item_col].value_counts().keys()[0]
    return best_seller
