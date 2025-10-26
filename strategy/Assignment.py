import numpy as np

def role_assignment(teammate_positions, formation_positions): 

    # Input : Locations of all teammate locations and positions
    # Output : Map from unum -> positions
    #-----------------------------------------------------------

    players_preferences = {i: [] for i in range(1, 6)}
    roles_preferences = {i: [] for i in range(1, 6)}

    # Setting preferences based on closest euclidean distance

    for i in range (0, 5):
        teammate_position_curr = teammate_positions[i]
        x_1 = teammate_position_curr[0]
        y_1 = teammate_position_curr[1]

        d = {i: 0 for i in range(1, 6)} 

        for j in range (0, 5):
                formation_position_curr = formation_positions[j]
                x_2 = formation_position_curr[0]
                y_2 = formation_position_curr[1]
            
                d[j+1] = np.sqrt(np.power(x_1-x_2, 2) + np.power(y_1-y_2, 2))

        players_preferences[i + 1] = sorted(d, key=d.get)  

    for i in range (0, 5):
        formation_position_curr = formation_positions[i]
        x_2 = formation_position_curr[0]
        y_2 = formation_position_curr[1]

        d = {i: 0 for i in range(1, 6)} 

        for j in range (0, 5):
                teammate_position_curr = teammate_positions[j]
                x_1 = teammate_position_curr[0]
                y_1 = teammate_position_curr[1]
                
                d[j+1] = np.sqrt(np.power(x_1-x_2, 2) + np.power(y_1-y_2, 2))

        roles_preferences[i + 1] = sorted(d, key=d.get)  
    
    # Pairing up players and Roles

    unmatched_players = [1, 2, 3, 4, 5]
    unmatched_roles = [1, 2, 3, 4, 5]

    point_preferences = {}

    while unmatched_players:
        player = unmatched_players[0]
        candidate = players_preferences[player]  

        for role in candidate:
            if role in unmatched_roles:
                formation_position_curr = formation_positions[role - 1]

                x = formation_position_curr[0]
                y = formation_position_curr[1]

                point_preferences[player] = ([x,y])
                unmatched_roles.remove(role)
                unmatched_players.pop(0)
                break

            elif role not in unmatched_roles:
                
                 current_match = None
                 for p, pos in point_preferences.items():
                     formation_position_curr = formation_positions[role - 1]
                     if pos[0] == formation_position_curr[0] and pos[1] == formation_position_curr[1]:
                         current_match = p
                         break
                 
                 role_candidates = roles_preferences[role]
                 if role_candidates.index(player) < role_candidates.index(current_match):
                     formation_position_curr = formation_positions[role - 1]
                     x = formation_position_curr[0]
                     y = formation_position_curr[1]
                     point_preferences[player] = ([x, y])
                     unmatched_players.pop(0)

                     del point_preferences[current_match]
                     unmatched_players.insert(0, current_match)
                     break 
    

    return point_preferences