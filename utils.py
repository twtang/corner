from fastai.tabular.all import * 

def joinLastGamesStatsHomeAway(df, stats, window=5):
    if not isinstance(stats, list): stats = [stats]
    dfs_home, dfs_away = [], []
    for stat in stats:        
        if stat in ['FTHG', 'HTHG', 'HS', 'HST', 'HC']:
            df_home = df.sort_values(['HomeTeam', 'Date']).reset_index(drop=True)
            df_home = df_home.set_index('Date').groupby('HomeTeam')[stat].rolling(window=window, min_periods=1).mean().reset_index()
            df_home = df_home.set_index(['Date', 'HomeTeam']).groupby('HomeTeam')[stat].shift(1).reset_index()
            df_home = df_home.rename(columns={f'{stat}':f'{stat}Last{window}Avg'})
            dfs_home.append(df_home)
            
        elif  stat in ['FTAG', 'HTAG', 'AS', 'AST', 'AC']:
            df_away = df.sort_values(['AwayTeam', 'Date']).reset_index(drop=True)
            df_away = df_away.set_index('Date').groupby('AwayTeam')[stat].rolling(window=window, min_periods=1).mean().reset_index()
            df_away = df_away.set_index(['Date', 'AwayTeam']).groupby('AwayTeam')[stat].shift(1).reset_index()
            df_away = df_away.rename(columns={f'{stat}':f'{stat}Last{window}Avg'})
            dfs_away.append(df_away)
        
    dfs_home = reduce(lambda left,right: pd.merge(left, right), dfs_home)
    dfs_away = reduce(lambda left,right: pd.merge(left, right), dfs_away)
    return dfs_home, dfs_away

def joinLastGamesStatsForAgainst(df, stats, window=5):
    if not isinstance(stats, list): stats = [stats]
    dfs_for, dfs_against = [], []
    for stat in stats:    
        # For
        df_home = df.sort_values(['HomeTeam', 'Date']).reset_index(drop=True)
        df_away = df.sort_values(['AwayTeam', 'Date']).reset_index(drop=True)
        df_home = df_home[['HomeTeam', 'Date', stat[0]]].rename(columns={'HomeTeam':'Team', stat[0]:stat[2]})
        df_away = df_away[['AwayTeam', 'Date', stat[1]]].rename(columns={'AwayTeam':'Team', stat[1]:stat[2]})
        df_for = pd.concat([df_home, df_away]).sort_values(['Team', 'Date'])
        df_for = df_for.set_index('Date').groupby('Team')[stat[2]].rolling(window=window, min_periods=1).mean().reset_index()
        df_for = df_for.set_index(['Date', 'Team']).groupby('Team')[stat[2]].shift(1).reset_index()
        df_for = df_for.rename(columns={stat[2]:stat[2]+f'ForLast{window}Avg'})
        dfs_for.append(df_for)
        
        # Against
        df_home = df.sort_values(['HomeTeam', 'Date']).reset_index(drop=True)
        df_away = df.sort_values(['AwayTeam', 'Date']).reset_index(drop=True)
        df_home = df_home[['HomeTeam', 'Date', stat[1]]].rename(columns={'HomeTeam':'Team', stat[1]:stat[2]})
        df_away = df_away[['AwayTeam', 'Date', stat[0]]].rename(columns={'AwayTeam':'Team', stat[0]:stat[2]})
        df_against = pd.concat([df_home, df_away]).sort_values(['Team', 'Date'])
        df_against = df_against.set_index('Date').groupby('Team')[stat[2]].rolling(window=window, min_periods=1).mean().reset_index()
        df_against = df_against.set_index(['Date', 'Team']).groupby('Team')[stat[2]].shift(1).reset_index()
        df_against = df_against.rename(columns={stat[2]:stat[2]+f'AgainstLast{window}Avg'})
        dfs_against.append(df_against)
        
    dfs_for = reduce(lambda left,right: pd.merge(left, right), dfs_for)
    dfs_against = reduce(lambda left,right: pd.merge(left, right), dfs_against)
    return dfs_for, dfs_against