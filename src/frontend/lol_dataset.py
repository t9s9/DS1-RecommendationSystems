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
import src.frontend.page_handling as page_handling
from surprise.model_selection import GridSearchCV
from .SessionState import session_get
from .util import timer
from src.frontend.dataset import DatasetWrapper
import os
import math

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

"""Taken from https://stackoverflow.com/a/16696317"""
def download_file(url,local):
    local_filename = local
    prog = st.progress(0)
    st.write("Downloading database from public URL")
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True,verify=False) as r:
        r.raise_for_status()
        total_size = 0
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65565): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
                total_size+=len(chunk)
                prog.progress(int(total_size*100/1000000000))
    prog.progress(100)
    return local_filename


@timer
def app():
    with st.sidebar:
        if st.button("Back"):
            page_handling.handler.set_page("menu")
            return
     
    state = session_get()
    st.title("The dataset")
    st.write("The dataset was created by crawling through the official Game API. Some minimal filtering have been applied to the dataset to clean it up and increase significance. Since the game is subject to frequent patches and balance updates, the data is highly volatile and significance declines rather fast. Therefore only games played in the last 100 days have been considered as input.")

    if not os.path.exists("src/lol_dataset/league_of_legends.db"):
        download_file("https://www.kava-i.de/league_of_legends.db","src/lol_dataset/league_of_legends.db")

    connection = sql.connect("src/lol_dataset/league_of_legends.db")
    cursor = connection.cursor()
    total_games = 0
    max_game_length = 0
    min_game_creation = 0


    for row in cursor.execute("SELECT MAX(game_duration) from matches"):
        max_game_length = row[0]

    for row in cursor.execute("SELECT MIN(creation_time) from matches"):
        min_game_creation = row[0]

    min_game_creation = -math.floor((int(time.time()*1000) - min_game_creation)/(1000*60*60*24))-1

    ok_side_form = st.sidebar.form(key='my_form')
    game_length_min = ok_side_form.slider("Select the game length in minutes:",min_value=0,max_value=int(max_game_length/60),value=(0,10))
    sample_range_days = ok_side_form.slider("Select the sample range in days:",min_value=min_game_creation,max_value=0,value=(min_game_creation,0))

    multi = ok_side_form.multiselect("Selects the region in the world from which to sample the games",["EU West","EU Nord", "Nord America", "Russia","Latein America 1","Latein America 2"],["EU West","EU Nord", "Nord America", "Russia","Latein America 1","Latein America 2"],key="Unique3")

    inv_mappings = { "https://euw1.api.riotgames.com":"EU West", "https://eun1.api.riotgames.com": "EU Nord" , "https://na1.api.riotgames.com":"Nord America", "https://la1.api.riotgames.com":"Latein America 1", "https://la2.api.riotgames.com":"Latein America 2","https://ru.api.riotgames.com":"Russia"}

    for row in cursor.execute("SELECT COUNT(champion_id) from matches"):
        total_games = row[0]

    epoch_time = int(time.time()*1000)


    mappings = {"EU West": "https://euw1.api.riotgames.com", "EU Nord": "https://eun1.api.riotgames.com", "Nord America": "https://na1.api.riotgames.com", "Latein America 1":"https://la1.api.riotgames.com", "Latein America 2": "https://la2.api.riotgames.com","Russia":"https://ru.api.riotgames.com"}

    execute_string = '('
    for x in multi:
        if execute_string != '(':
            execute_string+=" OR "
        execute_string += "idx = \""+mappings[x]+"\""
    execute_string+=")"


    filtered = 0
    cursor.execute("DROP VIEW IF EXISTS filtered_matches")
    cursor.execute("CREATE VIEW filtered_matches AS SELECT * from matches WHERE game_duration >= {} AND game_duration <= {} AND creation_time >= {} AND creation_time <= {} AND {}".format(game_length_min[0]*60,game_length_min[1]*60,(epoch_time+sample_range_days[0]*60*60*24*1000),epoch_time+sample_range_days[1]*60*60*24*1000,execute_string))


    data = []
    data_2 = []
    for row in cursor.execute("SELECT idx,COUNT(*) from filtered_matches GROUP BY idx"):
        data.append(row[1])
        data_2.append(inv_mappings[row[0]])
    region = pd.DataFrame(data,index=data_2,columns=["Data points"])

    st.bar_chart(region)

    js = json.loads(open("data/lol_dataset/champion.json","r").read())
    champions = []
    champion_dict = {}
    for x in js["data"]:
        champions.append(x)
        champion_dict[x] = (int(js["data"][x]["key"]),js["data"][x]["blurb"])

    data = []
    data_2 = []
    for row in cursor.execute("SELECT champion_id,COUNT(*) from filtered_matches GROUP BY champion_id ORDER BY Count(*) DESC LIMIT 20"):
        data.append(row[1])
        for x in champion_dict:
            if row[0] == champion_dict[x][0]:
                data_2.append(x)

    champs = pd.DataFrame(data,index=data_2,columns=["Data points"])
    st.bar_chart(champs)

    data = []
    data_2 = []
    for row in cursor.execute("SELECT champion_id,COUNT(*) from filtered_matches GROUP BY champion_id ORDER BY Count(*) ASC LIMIT 20"):
        data.append(row[1])
        for x in champion_dict:
            if row[0] == champion_dict[x][0]:
                data_2.append(x)

    champs = pd.DataFrame(data,index=data_2,columns=["Data points"])
    st.bar_chart(champs)


    for row in cursor.execute("SELECT COUNT(champion_id) from filtered_matches"):
        st.write("A total of ",row[0]," games have been found in the database with the given filters.")
        filtered = row[0]


    st.write("This is ",round(100*filtered/total_games,2),"% of the whole dataset.")


    st.title("Showcase")
    col1, col2, col3 = st.beta_columns([1,1,1])

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


    own_champ = int(champion_dict[options][0])
    other_champ = int(champion_dict[options2][0])

    counter = 0
    anti_counter = 0

    start_time = 0
    duration = 0
    prog = st.progress(0)
    col1,col2, col3 = st.beta_columns([1,4,1])
    frames = {"userID": [], "itemID": [], "rating": []}
    winning_items = {}
    running_counter = 0
    with col2.empty():
        prog.progress(0)
        for row in cursor.execute("SELECT e.champion_id, e.items FROM filtered_matches e INNER JOIN filtered_matches m ON m.game_id = e.game_id WHERE e.win = 1 AND m.win=0 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
            counter+=1
            items = json.loads(row[1])
            coldcase_prediction([items["item0"],items["item1"],items["item2"],items["item3"],items["item4"],items["item5"],items["item6"]],winning_items,1,all_items)

        for row in cursor.execute("SELECT m.champion_id, e.items FROM filtered_matches e INNER JOIN filtered_matches m ON m.game_id = e.game_id WHERE e.win = 0 AND m.win=1 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
            anti_counter+=1
            items = json.loads(row[1])
            coldcase_prediction([items["item0"],items["item1"],items["item2"],items["item3"],items["item4"],items["item5"],items["item6"]],winning_items,0,all_items)

    prog.progress(100)

    val = 0
    if (counter+anti_counter) != 0:
        winning_list = []
        for x in winning_items:
            if winning_items[x][1] >= val:
                if not is_item(x,item_dict):
                    continue
                estimated_p = winning_items[x][0]/winning_items[x][1]
                cost = item_dict[str(x)]["gold"]["total"]
                winning_list.append((estimated_p,x,cost))

        games = []

        for x in range(counter):
            games.append(1)
        for x in range(anti_counter):
            games.append(0)

        list_items_filtered = []
        list_items_names = []
        for x in winning_list:
            list_items_filtered.append(x[0])
            list_items_names.append(item_dict[str(x[1])]["name"])
        outcomes = np.array(games)

        
        name = ok_side_form.text_input("Name for the dataset:",key="Unique4")
        submit_button2x = ok_side_form.form_submit_button(label='Apply')
        submit_button = st.sidebar.button(label='Add to dataset')
        if submit_button:
            running_counter = 0
            for row in cursor.execute("SELECT DISTINCT e.items, e.summoner_name, e.game_id FROM matches e INNER JOIN matches m ON m.game_id = e.game_id WHERE e.win = 1 AND m.win=0 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
                running_counter+=1
                items = json.loads(row[0])
                add_item_full(frames,items,str(running_counter)+" | "+row[1],True,winning_items,val,item_dict)
            for x in range(len(frames["itemID"])):
                idx = item_dict[str(frames["itemID"][x])]["name"]
                frames["itemID"][x] = idx
                
            for x in range(len(frames["userID"])):
                idx = item_dict[str(frames["itemID"][x])]["name"]
                frames["itemID"][x] = idx


            df = pd.DataFrame(frames)
            print(df)


            dataset = DatasetWrapper(name=name, data=df,id=1, param={"own_champ": options,"enemy_champ": options2, "item_limit": 0, "poor_mans_choice": False})
            state.datasets.append(dataset)
            st.sidebar.success(f"Dataset '{name}' saved.")

        wilson = confidence_wilson_score(counter/(counter+anti_counter),counter+anti_counter)
        st.latex("P(X=(win,"+options+") \land Y=(lose,"+options2+")) = \\frac{1}{N}\\cdot\\sum_{i=0}^{N}{x_i}="+"{:.2f}".format(counter/(counter+anti_counter)))
        st.markdown("**Confidence interval based on Wilson Score:**")
        st.latex("P\\left(\\Theta\mid y \leq \\frac{p-\\Theta}{\\sqrt{\\frac{1}{n}\cdot p \cdot (1-p)}} \leq z_{0.90}\\right) = "+str(confidence_wilson_score(counter/(counter+anti_counter),counter+anti_counter)))
        st.write("Based on the above figures the average winrate is ",round(counter/(counter+anti_counter),2)," with a 90% confidence interval of ",(round(wilson[0],2),round(wilson[1],2))," with a sample size of ",counter+anti_counter)
    else:
        st.error("Could not find any games with this matchup")



