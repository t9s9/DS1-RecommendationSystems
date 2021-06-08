import pickle
import pandas


loads = pandas.read_csv("../../data/lol_dataset/archive/match_data_version1.csv")
for x in loads.participants:
    team_id = -1
    for entry in x:
        if hasattr(entry,'stats'):
            if team_id == -1:
                team_id = entry.teamId
            if hasattr(entry.stats,'win'):


