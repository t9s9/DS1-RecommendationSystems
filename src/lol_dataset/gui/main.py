import streamlit as st
import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
from tqdm import trange
import time
import numpy as np
from scipy import stats
from math import sqrt
from surprise import SVD
from surprise import NMF
from surprise.prediction_algorithms.knns import KNNBaseline
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.model_selection import GridSearchCV

def confidence_wilson_score2(wins, loses):
    n = wins + loses 

    if n == 0:
        return 0

    z = 1.44 # 95% target error
    phat = float(wins) / n
    return ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))

def is_item(item,all_items):
    if item == 0:
        return False
    if "into" in all_items[str(item)]:
        return all_items[str(item)]["into"] == []
    if all_items[str(item)]["gold"]["purchasable"] == True:
        return all_items[str(item)]["gold"]["total"] != 0
    return False

def confidence_wilson_score(p, n, z = 1.44):
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (lower_bound, upper_bound)

def add_item(lst,lst2,lst3,itm,counter,rating):
    if itm != 0:
        lst.append(itm)
        lst2.append(counter)
        lst3.append(rating)

def add_item_full(lst,items,counter,win,win_items,thresh,all_items):
    for x in ["item0","item1","item2","item3","item4","item5","item6"]:
        if not is_item(items[x],all_items):
            continue
        if win_items[items[x]][1] > thresh:
            add_item(lst["itemID"],lst["userID"],lst["rating"],items[x],counter,1-(win_items[items[x]][0]/win_items[items[x]][1]))
        else:
            add_item(lst["itemID"],lst["userID"],lst["rating"],items[x],counter,1)


def get_leading(average_win_rate,p):
    if average_win_rate < p:
        return 1
    return -1



def coldcase_prediction(items,current_dict,rating,all_items):
    for x in items:
        if x != 0:
            if x in current_dict:
                current_dict[x] = (current_dict[x][0]+rating,current_dict[x][1]+1)
            else:
                current_dict[x] = (rating,1)




st.title("League of legends - Modern E-Gaming")
st.write("E-Gaming is one of the fastest growing areas of the economy. The revenue is growing very fast and there is no sign of slowing down.")

df = pd.DataFrame([869,1184,1592,2173,2963],
                    columns=["Revenue in million $"],
                   index=[int(2018),int(2019),int(2020),int(2021),int(2022)])
st.line_chart(df)

st.write("Competetive tournaments raise an evergrowing prize pool the League of Legends World Championsship 2018 has a prize pool of $6,450,000.")
st.image("http://img2.chinadaily.com.cn/images/201807/17/5b4d2be6a310796d8b4c0773.jpeg",caption="The King Pro Leaguetournament in Shanghai")

st.write("The dataset was chosen because the significance of machine learning in E-Gaming is rising with every year. To improve the understanding and maybe shed light on the complex dynamics which are involved in playing such a game I chose this dataset.")

st.title("The dataset")
st.write("The dataset was created by crawling through the official Game API. Some minimal filtering have been applied to the dataset to clean it up and increase significance. Since the game is subject to frequent patches and balance updates, the data is highly volatile and significance declines rather fast. Therefore only games played in the last 100 days have been considered as input.")
connection = sql.connect("../league_of_legends.db")
cursor = connection.cursor()
total_games = 0
max_game_length = 0
min_game_creation = 0
for row in cursor.execute("SELECT COUNT(champion_id) from matches"):
    st.write("A total of ",row[0]," games have been found in the database with the given filters.")
    total_games = row[0]

st.write("This is ",100,"% of the whole dataset.")

for row in cursor.execute("SELECT MAX(game_duration) from matches"):
    max_game_length = row[0]

for row in cursor.execute("SELECT MIN(creation_time) from matches"):
    min_game_creation = row[0]

min_game_creation = -int((int(time.time()*1000) - min_game_creation)/(1000*60*60*24))

st.title("Showcase")
col1, col2, col3 = st.beta_columns([1,1,1])
col2.button("Reset showcase")

js = json.loads(open("../../../data/lol_dataset/champion.json","r").read())
champions = []
champion_dict = {}
for x in js["data"]:
    champions.append(x)
    champion_dict[x] = (int(js["data"][x]["key"]),js["data"][x]["blurb"])

create_constrains = ""
col1, col3,col2 = st.beta_columns([3,1,3])
options = col1.selectbox('Select your champion',champions)
options2 = col2.selectbox('Select the enemy champion',champions)

lore_url = "http://ddragon.leagueoflegends.com/cdn/11.11.1/data/en_US/champion/{}.json".format(options)
lore_enemy_url = "http://ddragon.leagueoflegends.com/cdn/11.11.1/data/en_US/champion/{}.json".format(options2)

lore_own = json.loads(requests.get(lore_url).text)
lore_enemy = json.loads(requests.get(lore_enemy_url).text)
all_items = requests.get('http://ddragon.leagueoflegends.com/cdn/11.11.1/data/en_US/item.json').json()["data"]
item_dict = {}
all_item_names = []
for i in all_items:
    all_items[i]["base_id"] = i
    item_dict[i] = all_items[i]
    item_dict[all_items[i]["name"]] = all_items[i]
    all_item_names.append(all_items[i]["name"])


col1.image("http://ddragon.leagueoflegends.com/cdn/img/champion/loading/{}_0.jpg".format(options),use_column_width="always")
with col1.beta_expander("See hero description"):
    st.write(lore_own["data"][options]["lore"])

col2.image("http://ddragon.leagueoflegends.com/cdn/img/champion/loading/{}_0.jpg".format(options2),use_column_width="always")
with col2.beta_expander("See hero description"):
    st.write(lore_enemy["data"][options2]["lore"])
with st.sidebar.form(key='my_form'):
    side = st.slider("Select the game length in minutes:",min_value=0,max_value=int(max_game_length/60),value=(0,10))
    side2 = st.slider("Select the sample range in days:",min_value=min_game_creation,max_value=0,value=(min_game_creation,0))
    multi = st.multiselect("Selects the region in the world from which to sample the games",["EU West","EU Nord", "Nord America", "Russia","Latein America 1","Latein America 2"],["EU West","EU Nord", "Nord America", "Russia","Latein America 1","Latein America 2"])
    submit_button = st.form_submit_button(label='Submit')
    
own_champ = int(champion_dict[options][0])
other_champ = int(champion_dict[options2][0])

counter = 0
anti_counter = 0

start_time = 0
duration = 0
prog = st.progress(0)
col1,col2, col3 = st.beta_columns([1,4,1])
frames = {"itemID": [], "userID": [], "rating": []}
winning_items = {}
running_counter = 0
with col2.empty():
    prog.progress(0)
    for row in cursor.execute("SELECT e.champion_id, e.items FROM matches e INNER JOIN matches m ON m.game_id = e.game_id WHERE e.win = 1 AND m.win=0 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
        counter+=1
        items = json.loads(row[1])
        coldcase_prediction([items["item0"],items["item1"],items["item2"],items["item3"],items["item4"],items["item5"],items["item6"]],winning_items,1,all_items)

    for row in cursor.execute("SELECT m.champion_id, e.items FROM matches e INNER JOIN matches m ON m.game_id = e.game_id WHERE e.win = 0 AND m.win=1 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
        anti_counter+=1
        items = json.loads(row[1])
        coldcase_prediction([items["item0"],items["item1"],items["item2"],items["item3"],items["item4"],items["item5"],items["item6"]],winning_items,0,all_items)
    
prog.progress(100)
if (counter+anti_counter) != 0:
    games = []

    for x in range(counter):
        games.append(1)
    for x in range(anti_counter):
        games.append(0)

    outcomes = np.array(games)

    wilson = confidence_wilson_score(counter/(counter+anti_counter),counter+anti_counter)
    st.latex("P(X=(win,"+options+") \land Y=(lose,"+options2+")) = \\frac{1}{N}\\cdot\\sum_{i=0}^{N}{x_i}="+"{:.2f}".format(counter/(counter+anti_counter)))
    st.markdown("**Confidence interval based on Wilson Score:**")
    st.latex("P\\left(\\Theta\mid y \leq \\frac{p-\\Theta}{\\sqrt{\\frac{1}{n}\cdot p \cdot (1-p)}} \leq z_{0.90}\\right) = "+str(confidence_wilson_score(counter/(counter+anti_counter),counter+anti_counter)))
    st.write("Based on the above figures the average winrate is ",round(counter/(counter+anti_counter),2)," with a 90% confidence interval of ",(round(wilson[0],2),round(wilson[1],2))," with a sample size of ",counter+anti_counter)
    
    selects = st.multiselect("Select the items you have already bought",all_item_names)
    average_win_rate = counter/(counter+anti_counter)


    running_counter+=1

     
    default = int((1/13)*(counter+anti_counter))
    val = st.slider(label="Select min sample size for calculation",min_value=0,max_value=counter+anti_counter,value=default)
    poor_mans_choice = st.checkbox("Activate poor mans choice")

    winning_list = []
    for x in winning_items:
        if winning_items[x][1] >= val:
            if not is_item(x,item_dict):
                continue
            estimated_p = winning_items[x][0]/winning_items[x][1]
            cost = item_dict[str(x)]["gold"]["total"]
            winning_list.append((estimated_p,x,cost))
    if len(selects) == 0:
        st.markdown("### Coldcase prediction")
        st.write("Predicting by taking absolute winning rates associated with this matchup.")

        if poor_mans_choice:
            winning_list.sort(key=lambda x: get_leading(average_win_rate,x[0])*(x[0]-average_win_rate)**2/(x[2]+1),reverse=True)
        else:
            winning_list.sort(key=lambda x: x[0],reverse=True)

        list_items_filtered = []
        list_items_names = []
        for x in winning_list:
            list_items_filtered.append(x[0])
            list_items_names.append(item_dict[str(x[1])]["name"])

        df = pd.DataFrame(list_items_filtered,
                    columns=["Ratings for every item"],
                   index=list_items_names)
        st.bar_chart(df,use_container_width=True)
        df_rating = pd.DataFrame({"item": [item_dict[str(x[1])]["name"] for x in winning_list],"win rate": ["{:.2f}%".format(x[0]*100) for x in winning_list],"sample size": [winning_items[x[1]][1] for x in winning_list], "effective score": ["{:.8f}".format(get_leading(average_win_rate,x[0])*(x[0]-average_win_rate)**2/(x[2]+1)) for x in winning_list]})
        st.table(df_rating)
    else:
        running_counter = 0
        for row in cursor.execute("SELECT e.champion_id, e.items FROM matches e INNER JOIN matches m ON m.game_id = e.game_id WHERE e.win = 1 AND m.win=0 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
            running_counter+=1
            items = json.loads(row[1])
            add_item_full(frames,items,running_counter,True,winning_items,val,item_dict)
        st.write(selects)
        item_dict_user = {}
        for x in ["item0","item1","item2","item3","item4","item5","item6"]:
            item_dict_user[x] = 0

        for x in zip(selects,["item0","item1","item2","item3","item4","item5","item6"]):
            item_dict_user[x[1]] = int(item_dict[x[0]]["base_id"])

        add_item_full(frames,item_dict_user,running_counter,True,winning_items,val,item_dict)
        
        df = pd.DataFrame(frames)

        # A reader is still needed but only the rating_scale param is requiered.
        reader = Reader(rating_scale=(0, 1))
        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        train = data.build_full_trainset()

        data_x = []
        data_rmse = []
        data_mae = []
        #for x in range(50,100,10):
            #algo = KNNBaseline(sim_options={"user_based": True, "name":"pearson"},k=x,n_epochs=x)
        #    algo = KNNWithMeans(sim_options={"user_based": True, "name":"pearson"},k=x,min_k=5)
            # algo = SVD(n_epochs=x)
        #    result = cross_validate(algo,data,cv=3,return_train_measures=True,verbose=True)
        #    data_x.append(x)
        #    data_rmse.append(np.mean(result["test_rmse"]))
        #    data_mae.append(np.mean(result["test_mae"]))

        #df_data = pd.DataFrame(data_rmse,
        #            columns=["RMSE for different k"],
        #           index=data_x)

        #st.line_chart(df_data)

        #df_data = pd.DataFrame(data_mae,
        #            columns=["MAE for different k"],
        #           index=data_x)

        #st.line_chart(df_data)

        algo = KNNWithMeans(sim_options={"user_based": True, "name":"pearson"},k=15,min_k=5)
        algo.fit(train)
        ratings = []
        #param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],'reg_all': [0.4, 0.6]}
        #gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3)

        #print(gs.fit(data))
        #print(gs)
        print(running_counter)
        
        list_items_filtered = []
        list_items_names = []
        for x in train.all_items():
            raw = train.to_raw_iid(x)
            if (item_dict[str(raw)]["name"] not in selects) and is_item(raw,item_dict):
                if winning_items[raw][1] < val:
                    continue
                pred = algo.predict(running_counter,train.to_raw_iid(x))
                cost = item_dict[str(raw)]["gold"]["total"]
                ratings.append((pred[3],raw,cost))

                list_items_filtered.append(pred[3])
                list_items_names.append(item_dict[str(ratings[-1][1])]["name"])

        df = pd.DataFrame(list_items_filtered,
                    columns=["Ratings for every item"],
                   index=list_items_names)
        st.bar_chart(df,use_container_width=True)


        print(ratings)
        if poor_mans_choice:
            ratings.sort(key=lambda x: get_leading(average_win_rate,1-x[0])*((1-x[0])-average_win_rate)**2/(x[2]+1),reverse=True)
        else:
            ratings.sort(key=lambda x: (1-x[0]),reverse=True)

        print(ratings)
        print("Nice one :D")

        df_rating = pd.DataFrame({"item": [item_dict[str(x[1])]["name"] for x in ratings],"win rate": ["{:.2f}%".format((1-x[0])*100) for x in ratings],"sample size": [winning_items[x[1]][1] for x in ratings],"effective score": ["{:.8f}".format(get_leading(average_win_rate,1-x[0])*((1-x[0])-average_win_rate)**2/(x[2]+1)) for x in ratings]})
        st.table(df_rating)
else:
    st.error("Could not find any games with this matchup")

