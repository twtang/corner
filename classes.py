from datetime import datetime

class Team():
    
    id = ''
    team_name=''
    
    def __init__(self, id, team_name):
        
        self.id = id
        self.team_name = team_name
        
    def __repr__(self):
        
        return f'{self.team_name} [{self.id}]'.replace('(', '').replace(')', '').replace(',', '')
    
class League():
    
    id = ''
    short_name = ''
    name = ''
    
    def __init__(self, id, short_name, name):
        
        self.id = id
        self.short_name = short_name
        self.name = name
        
    def __repr__(self):
        
        return f'{self.name} [{self.short_name}]'.replace('(', '').replace(')', '').replace(',', '')
    
class Match():
    


    def had_odds_handler(self, data):

        if data is None:

            return dict(HAD_H='', HAD_D='', HAD_A='')

        return dict(HAD_H=data['H'][4:], HAD_D=data['D'][4:], HAD_A=data['A'][4:])

    def ooe_odds_handler(self, data):

        if data is None:

            return dict(OOE_ODD='', OOE_EVEN='')

        return dict(OOE_ODD=data['O'][4:], OOE_EVEN=data['E'][4:])

    def ttg_odds_handler(self, data):

        if data is None:

            return dict(TTG_0='', TTG_1='', TTG_2='', TTG_3='', TTG_4='', TTG_5='', TTG_6='', TTG_7='')

        return dict(
            TTG_0=data['P0'][4:], 
            TTG_1=data['P1'][4:], 
            TTG_2=data['P2'][4:], 
            TTG_3=data['P3'][4:], 
            TTG_4=data['P4'][4:], 
            TTG_5=data['P5'][4:], 
            TTG_6=data['P6'][4:], 
            TTG_7=data['M7'][4:])

    def hft_odds_handler(self, data):

        if data is None:

            return dict(
                HFT_HH='', HFT_HD='', HFT_HA='', 
                HFT_DH='', HFT_DD='', HFT_DA='', 
                HFT_AH='', HFT_AD='', HFT_AA='')

        return dict(
            HFT_HH=data['HH'][4:], HFT_HD=data['HD'][4:], HFT_HA=data['HA'][4:], 
            HFT_DH=data['DH'][4:], HFT_DD=data['DD'][4:], HFT_DA=data['DA'][4:], 
            HFT_AH=data['AH'][4:], HFT_AD=data['AD'][4:], HFT_AA=data['AA'][4:])

    def hha_odds_handler(self, data):

        if data is None:

            return dict(
                HHA_HG='', HHA_AG='',
                HHA_H='', HHA_D='', HHA_A='')

        return dict(
            HHA_HG=data['HG'], HHA_AG=data['AG'],
            HHA_H=data['H'][4:], HHA_D=data['D'][4:], HHA_A=data['A'][4:])

    def hdc_odds_handler(self, data):

        if data is None:

            return dict(
                HDC_HG='', HDC_AG='',
                HDC_H='', HDC_D='', HDC_A='')

        return dict(
            HDC_HG=data['HG'], HDC_AG=data['AG'],
            HDC_H=data['H'][4:],HDC_A=data['A'][4:])

    def hil_odds_handler(self, data):

        if data is None:

            return dict(
                HIL_LINE='', HIL_H='', HIL_L=''
            )

        return dict(
            HIL_LINE=data['LINELIST'][0]['LINE'], 
            HIL_H=data['LINELIST'][0]['H'][4:], 
            HIL_L=data['LINELIST'][0]['L'][4:]
        )

#     def chl_odds_handler(self, data):

#         if data is None:

#             return dict(
#                 CHL_LINE='', CHL_H='', CHL_L=''
#             )

#         return dict(
#             CHL_LINE=data['LINELIST'][0]['LINE'], 
#             CHL_H=data['LINELIST'][0]['H'][4:], 
#             CHL_L=data['LINELIST'][0]['L'][4:]
#         )
    
    def chl_odds_handler(self, data):

        if data is None:

            return dict(
                MAINLINE='', CHL_LINE_0='', CHL_H_0='', CHL_L_0=''
            )

        v = [[line['MAINLINE'], line['LINE'].split('/')[0], line['H'][4:], line['L'][4:]] for line in data['LINELIST']]
        k = [[f'MAINLINE_{i}', f'CHL_LINE_{i}', f'CHL_H_{i}', f'CHL_L_{i}'] for i in range(len(data['LINELIST']))]
        v = [item for sublist in v for item in sublist]
        k = [item for sublist in k for item in sublist]
        
        return dict(zip(k, v))
    
    def __init__(self, m):
        self.date = None
        self.time = None
        self.id = None
        self.num = None
        self.short_id = None

        self.home_team = None
        self.away_team = None
        self.league = None
        self.events = None
        self.odds = {}
        
        self.league = League(
                m['tournament']['tournamentID'], 
                m['tournament']['tournamentShortName'], 
                m['tournament']['tournamentNameEN']
            )
        self.home_team = Team(
                m['homeTeam']['teamID'],
                m['homeTeam']['teamNameEN']
            )
        self.away_team = Team(
                m['awayTeam']['teamID'],
                m['awayTeam']['teamNameEN']
            )

        self.events = m['liveEvent']['hasLiveInfo']
        self.id = m['matchID']
        self.num = m['matchNum']
        # self.date = m['matchDate'].split('+')[0]
        # 2021-09-13T00:00:00+08:00
        self.date = m['matchTime'].split('T')[0]
        self.time = m['matchTime'].split('T')[1].split('.')[0]
        self.short_id = m['matchIDinofficial']

        odds_handlers = {
            # 'HAD': self.had_odds_handler,
            # 'FHA': self.fha_odds_handler,
            # 'HHA': self.hha_odds_handler,
            # 'HDC': self.hdc_odds_handler,
            # 'HIL': self.hil_odds_handler,
            'CHL': self.chl_odds_handler,
            # 'TTG': self.ttg_odds_handler,
            # 'OOE': self.ooe_odds_handler,
            # 'HFT': self.hft_odds_handler,
        }
        
        for o in odds_handlers:

            if f'{o.lower()}odds' in m:
                self.odds[o] = odds_handlers[o](m[f'{o.lower()}odds'])
            else:
                self.odds[o] = odds_handlers[o](None)

    def __str__(self):

        return f'{str(self.league)} {str(self.home_team)} vs {str(self.away_team)}'.replace('(', '').replace(')', '').replace(',', '')

    def export(self):

        all_odds = {}
        for i in self.odds:
            all_odds = dict(all_odds, **self.odds[i])
        # values = all_odds.values()
        values = [
            self.short_id, datetime.strftime(datetime.strptime(self.date, '%Y-%m-%d'), '%Y-%b-%d'), self.time, str(self.league), 
            str(self.home_team.team_name), str(self.away_team.team_name)]
        values.extend(all_odds.values())

        print(values)

        return values

    def export_keys(self):

        all_odds = {}
        for i in self.odds:
            all_odds = dict(all_odds, **self.odds[i])
        values = ['short_id', 'date', 'time', 'league', 'home_team', 'away_team']
        values.extend(all_odds.keys())

        print(values)

        return values