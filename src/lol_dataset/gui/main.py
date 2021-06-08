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

def confidence_wilson_score2(wins, loses):
    n = wins + loses 

    if n == 0:
        return 0

    z = 1.44 # 95% target error
    phat = float(wins) / n
    return ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))

def confidence_wilson_score(p, n, z = 1.44):
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (lower_bound, upper_bound)

st.title("League of legends - Modern E-Gaming")
st.write("E-Gaming is one of the fastest growing areas of the economy. The revenue is growing very fast and there is no sign of slowing down.")

df = pd.DataFrame([869,1184,1592,2173,2963],
                    columns=["Revenue in million $"],
                   index=[2018,2019,2020,2021,2022])
st.line_chart(df)

st.write("Competetive tournaments raise an evergrowing prize pool the League of Legends World Championsship 2018 has a prize pool of $6,450,000.")
st.image("http://img2.chinadaily.com.cn/images/201807/17/5b4d2be6a310796d8b4c0773.jpeg",caption="The King Pro Leaguetournament in Shanghai")

st.write("The dataset was chosen because the significance of machine learning in E-Gaming is rising with every year. To improve the understanding and maybe shed light on the complex dynamics which are involved in playing such a game I chose this dataset.")

st.title("The dataset")
st.write("The dataset was created by crawling through the official Game API. Some minimal filtering have been applied to the dataset to clean it up and increase significance. Since the game is subject to frequent patches and balance updates, the data is highly volatile and significance declines rather fast. Therefore only games played in the last 100 days have been considered as input.")
connection = sql.connect("../crawler/league_of_legends.db")
cursor = connection.cursor()
total_games = 0
for row in cursor.execute("SELECT COUNT(champion_id) from matches"):
    st.markdown("## A total of "+str(row[0])+" games have been found in the database")
    total_games = row[0]

st.title("Showcase")
col1, col2, col3 = st.beta_columns([1,1,1])
col2.button("Reset showcase")

js = json.loads(open("../../../data/lol_dataset/champion.json","r").read())
champions = []
champion_dict = {}
for x in js["data"]:
    champions.append(x)
    champion_dict[x] = (int(js["data"][x]["key"]),js["data"][x]["blurb"])

multi = st.sidebar.multiselect("Selects the region in the world from which to sample the games",["EU West","EU Nord", "Nord America", "Russia","Latein America 1","Latein America 2"],["EU West","EU Nord", "Nord America", "Russia","Latein America 1","Latein America 2"])
create_constrains = ""
col1, col3,col2 = st.beta_columns([3,1,3])
options = col1.selectbox('Select your champion',champions)
options2 = col2.selectbox('Select the enemy champion',champions)

lore_url = "http://ddragon.leagueoflegends.com/cdn/11.11.1/data/en_US/champion/{}.json".format(options)
lore_enemy_url = "http://ddragon.leagueoflegends.com/cdn/11.11.1/data/en_US/champion/{}.json".format(options2)

lore_own = json.loads(requests.get(lore_url).text)
lore_enemy = json.loads(requests.get(lore_enemy_url).text)


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
with col2.empty():
    prog.progress(0)
    for row in cursor.execute("SELECT COUNT(e.champion_id) FROM matches e INNER JOIN matches m ON m.game_id = e.game_id WHERE e.win = 1 AND m.win=0 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
        counter=row[0]
    for row in cursor.execute("SELECT COUNT(m.champion_id) FROM matches e INNER JOIN matches m ON m.game_id = e.game_id WHERE e.win = 0 AND m.win=1 AND e.champion_id="+str(own_champ)+" AND m.champion_id="+str(other_champ)):
        anti_counter=row[0]
    
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
    st.latex("P\\left(\\Theta\mid y \leq \\frac{p-\\Theta}{\\sqrt{\\frac{1}{n}\cdot p \cdot (1-p)}} \leq z_{0.95}\\right) = "+str(confidence_wilson_score(counter/(counter+anti_counter),counter+anti_counter)))
    st.write("Based on the above figures the average winrate is ",round(counter/(counter+anti_counter),2)," with a 90% confidence interval of ",(round(wilson[0],2),round(wilson[1],2))," with a sample size of ",counter+anti_counter)
else:
    st.error("Could not find any games with this matchup")
