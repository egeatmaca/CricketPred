from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        team1 = request.form["team1"]
        team2 = request.form["team2"]
        return redirect(url_for("prediction", team1 = team1, team2 = team2))
    else:
        return render_template("index.html")

@app.route("/<team1>-<team2>")
def prediction(team1, team2):
    res, players1, players2 = icc_predict(team1, team2)
    return render_template("prediction.html", res = res, team1 = team1, team2 = team2, players1 = players1, players2 = players2)

#Match Result Prediction
country_percentages = {'australia': 73.936, 'india': 64.458, 'south africa': 61.905, 'new zealand': 60.674, 
                       'england': 59.639, 'pakistan': 58.442, 'west indies': 54.43, 'sri lanka': 50.625,
                       'bangladesh': 35.897, 'ireland': 35.714, 'kenya': 24.138, 'zimbabwe': 22.727,
                       'canada': 11.111, 'netherlands': 10, 'united arab emirates': 9.091, 'afghanistan': 6.667,
                       'bermuda': 0, 'scotland': 0, 'namibia': 0, 'east africa': 0}

results_data = pd.read_csv('match results data.csv')
X_res = results_data[['Team1', 'Team2']]
y_res = results_data['Winner']

lr_res = LogisticRegression()
lr_res.fit(X_res, y_res)

def predict_result(team1, team2):
    team1_p = country_percentages[team1.lower()]
    team2_p = country_percentages[team2.lower()]
    if team1_p > team2_p:
        if lr_res.predict([[team1_p, team2_p]])[0] == 0:
            return team1.upper() + " won"
        elif lr_res.predict([[team1_p, team2_p]])[0] == 1:
            return team2.upper() + " won"
    elif team2_p > team1_p:
        if lr_res.predict([[team2_p, team1_p]])[0] == 0:
            return team2.upper() + " won"
        elif lr_res.predict([[team2_p, team1_p]])[0] == 1:
            return team1.upper() + " won"
            
#Best 11's Prediction
player_data = pd.read_csv('player data.csv')
player_data['R/M'] = player_data['ODI runs'] / player_data['Matches']
player_data['B/M'] = player_data['Balls Bowled'] / player_data['Matches']
player_data['W/M'] = player_data['Wkts'] / player_data['Matches']
player_data['Win/Matches'] = player_data['won'] / player_data['Matches']

player_data['Role'] = player_data['Role'].str.lower()
player_data['country'] = player_data['country'].str.lower()

players = {}
for i in range(player_data.shape[0]):
    players.update({i: player_data['player name'][i]})

lr_player = LogisticRegression()

def select_players(team_name):
    player_types = {'batsman': [4, 'Matches', 'R/M', 'Average'], 'bowler': [4, 'Matches', 'B/M', 'W/M'], 
                    'all-rounder': [3, 'Matches', 'R/M', 'Average', 'B/M', 'W/M']}
    plyrs = []
    team_name = team_name.lower()
    if player_data.loc[player_data['country'] == team_name].loc[player_data['Role'] == 'wicket keeper'].shape[0] > 0:
        player_types.update({'bowler': [3, 'Matches', 'B/M', 'W/M']})
        player_types.update({'wicket keeper': [1, 'Matches', 'R/M', 'Average']})
    for player_type in player_types.keys():
        if player_type == 'allrounder':
            selected = player_data.loc[player_data['Role'] == 'batsman'].sort_values(ascending = False, by = 'Win/Matches').head(int(player_data.loc[player_data['Role'] == 'batsman'].shape[0]*0.1))[player_types['batsman'][1:len(player_types[player_type])]].append(
                       player_data.loc[player_data['Role'] == 'bowler'].sort_values(ascending = False, by = 'Win/Matches').head(int(player_data.loc[player_data['Role'] == 'bowler'].shape[0]*0.1))[player_types['bowler'][1:len(player_types[player_type])]])
            notselected = player_data.loc[player_data['Role'] == 'batsman'].sort_values(by = 'Win/Matches').head(int(player_data.loc[player_data['Role'] == 'batsman'].shape[0]*0.1))[player_types['batsman'][1:len(player_types['batsman'])]].append(
                        player_data.loc[player_data['Role'] == 'bowler'].sort_values(by = 'Win/Matches').head(int(player_data.loc[player_data['Role'] == 'bowler'].shape[0]*0.1))[player_types['bowler'][1:len(player_types['bowler'])]])
            selected = selected.fillna(0)
            notselected = notselected.fillna(0)
        else:
            selected = player_data.loc[player_data['Role'] == player_type].sort_values(ascending = False, by = 'Win/Matches').head(int(player_data.loc[player_data['Role'] == player_type].shape[0]*0.1))[player_types[player_type][1:len(player_types[player_type])]]
            notselected = player_data.loc[player_data['Role'] == player_type].sort_values(by = 'Win/Matches').head(int(player_data.loc[player_data['Role'] == player_type].shape[0]*0.1))[player_types[player_type][1:len(player_types[player_type])]]
        selected['Selected'] = 1
        notselected['Selected'] = 0
        player_type_train = selected.append(notselected)
        X_player = player_type_train[player_types[player_type][1:len(player_types[player_type])]]
        y_player = player_type_train['Selected']
        lr_player.fit(X_player , y_player)
        X = player_data.loc[player_data['Role'] == player_type].loc[player_data['country'] == team_name, player_types[player_type][1:len(player_types[player_type])]]
        pred = lr_player.predict_proba(X)
        pred = pd.DataFrame(data = pred, index = X.index, columns = ['worst', 'best']).sort_values(
               by = 'best', ascending = False)
        selected_players = pred.index[0:player_types[player_type][0]]
        plyrs.append(player_type.upper() + ':')
        for p in selected_players:
            plyrs.append(players[p])
    return plyrs
        
def icc_predict(team1, team2):
    return predict_result(team1, team2), select_players(team1), select_players(team2)

if __name__ == "__main__":
    app.run(debug = True)
