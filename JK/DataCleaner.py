import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from utility import *

homedir = get_homedir()

PATH_DEMO = f"{homedir}/data/us/aggregate_berkeley.csv"
PATH_MT = f"{homedir}/data/us/covid/nyt_us_counties.csv"
PATH_MB = f"{homedir}/data/us/mobility/DL-us-mobility-daterow.csv"
PATH_CLUSTER = f'{homedir}/JK/clustering/n_clusters=5_kmeans_extended.txt'
ON_CLUSTER = True

##################################################################################

FIPS_mapping, FIPS_full = get_FIPS(reduced=True)
oneweek = pd.Timedelta(days=7)
md_now = pd.Timestamp.now().strftime('%m%d')

"""
Read the datasets.
"""
berkeley = pd.read_csv(PATH_DEMO, index_col=0)
berkeley['countyFIPS'] = berkeley['countyFIPS'].apply(correct_FIPS)
berkeley = fix_FIPS(berkeley, fipslabel='countyFIPS', reduced=True)

popularity_type= ['PopMale<52010',
    'PopFmle<52010', 'PopMale5-92010', 'PopFmle5-92010', 'PopMale10-142010',
    'PopFmle10-142010', 'PopMale15-192010', 'PopFmle15-192010',
    'PopMale20-242010', 'PopFmle20-242010', 'PopMale25-292010',
    'PopFmle25-292010', 'PopMale30-342010', 'PopFmle30-342010',
    'PopMale35-442010', 'PopFmle35-442010', 'PopMale45-542010',
    'PopFmle45-542010', 'PopMale55-592010', 'PopFmle55-592010',
    'PopMale60-642010', 'PopFmle60-642010', 'PopMale65-742010',
    'PopFmle65-742010', 'PopMale75-842010', 'PopFmle75-842010',
    'PopMale>842010', 'PopFmle>842010']
popularity_type_Male = popularity_type[::2]
popularity_type_Fmle = popularity_type[1::2]
motality_type = ['3-YrMortalityAge<1Year2015-17',
    '3-YrMortalityAge1-4Years2015-17', '3-YrMortalityAge5-14Years2015-17',
    '3-YrMortalityAge15-24Years2015-17',
    '3-YrMortalityAge25-34Years2015-17',
    '3-YrMortalityAge35-44Years2015-17',
    '3-YrMortalityAge45-54Years2015-17',
    '3-YrMortalityAge55-64Years2015-17',
    '3-YrMortalityAge65-74Years2015-17',
    '3-YrMortalityAge75-84Years2015-17', '3-YrMortalityAge85+Years2015-17']

demo = pd.DataFrame()
demo['fips'] = berkeley['countyFIPS']
demo['PopRatioMale2017'] = berkeley['PopTotalMale2017'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['PopRatio65+2017'] = berkeley['PopulationEstimate65+2017'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['PopulationDensityperSqMile2010'] = berkeley['PopulationDensityperSqMile2010']
demo['MedicareEnrollment,AgedTot2017'] = berkeley['MedicareEnrollment,AgedTot2017'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['#Hospitals'] = 20000 * berkeley['#Hospitals'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['#ICU_beds'] = 10000 * berkeley['#ICU_beds'] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
for i in range(len(popularity_type_Male)):
    demo['PopRatio'+popularity_type_Male[i][3:]] = berkeley[popularity_type_Male[i]] / (berkeley[popularity_type_Male[i]]+berkeley[popularity_type_Fmle[i]])
    demo['PopRatio'+popularity_type_Male[i][7:]] = berkeley[popularity_type_Male[i]] / (berkeley['PopTotalMale2017']+berkeley['PopTotalFemale2017'])
demo['HeartDiseaseMortality'] = berkeley['HeartDiseaseMortality']
demo['StrokeMortality'] = berkeley['StrokeMortality']
demo['DiabetesPercentage'] = berkeley['DiabetesPercentage']
demo['Smokers_Percentage'] = berkeley['Smokers_Percentage']

demo.fillna(0, inplace=True)

motality = pd.read_csv(PATH_MT, parse_dates=['date'])
motality.dropna(inplace=True)
motality['fips'] = motality['fips'].apply(correct_FIPS)
motality = fix_FIPS(motality, fipslabel='fips', datelabel='date', reduced=True)

mobility = pd.read_csv(PATH_MB, parse_dates=['date'])
mobility.dropna(subset=['fips'], inplace=True)
mobility['fips'] = mobility['fips'].apply(correct_FIPS)
mobility.drop(columns=['admin_level', 'samples'], inplace=True)
mobility = fix_FIPS(mobility, fipslabel='fips', datelabel='date', reduced=True)

FIPS_demo = set(demo['fips']); FIPS_mt = set(motality['fips']); FIPS_mb = set(mobility['fips'])

date_st_mt = motality['date'].min(); date_ed_mt = motality['date'].max()
date_st_mb = mobility['date'].min(); date_ed_mb = mobility['date'].max()

"""
Filling in missing dates by searching closest date of the same day.
"""
ndays = (date_ed_mb - date_st_mb).days+1
dwin = pd.date_range(start=date_st_mb, end=date_ed_mb)
altrange = [item for sublist in [[n,-n] for n in range(1, ndays//7+1)] for item in sublist]

m50toAdd = []
for fips in FIPS_mb:
    df = mobility[mobility['fips']==fips]
    if len(df) != ndays:
        existingdates = list(df['date'])
        missingdates = set(dwin).difference(set(existingdates))
        for dt in missingdates:
            samedays = [dt + n*oneweek for n in altrange if (dt + n*oneweek) in existingdates]
            if samedays:
                m50, m50_index = df[df['date']==samedays[0]]['m50'].iloc[0], df[df['date']==samedays[0]]['m50_index'].iloc[0]
            else:
                m50, m50_index = df[df['date']==existingdates[-1]]['m50'].iloc[0], df[df['date']==existingdates[-1]]['m50_index'].iloc[0]
            m50toAdd.append([dt, fips, m50, m50_index])
mobility = mobility.append(pd.DataFrame(m50toAdd, columns=mobility.columns))

"""
Filling in missing counties using their state.
"""
for fips in FIPS_demo.difference(FIPS_mb):
    stt = str(int(fips[:2]))
    if stt in FIPS_mb:
        dummy = mobility[mobility['fips']==stt].copy()
        dummy.loc[:,'fips'] = fips
        mobility = mobility.append(dummy)
FIPS_mb = set(mobility['fips'])

"""
Save preprocessed dataframes.
"""
try:
    os.mkdir(f'{homedir}/JK/preprocessing/{md_now}')
except OSError as error:
    print(error)

demo.to_csv(f'{homedir}/JK/preprocessing/{md_now}/demographic.csv', index=False)
motality.to_csv(f'{homedir}/JK/preprocessing/{md_now}/motality.csv', index=False)
mobility.to_csv(f'{homedir}/JK/preprocessing/{md_now}/mobility.csv', index=False)

# settings
date_st = max(date_st_mt, date_st_mb)
date_ed = min(date_ed_mt, date_ed_mb)
date_win = pd.date_range(start=date_st, end=date_ed)

columns_demo = list(demo.columns); columns_demo.remove('fips')
columns_mt = ['cases', 'deaths']
columns_mb = ['m50', 'm50_index']
columns_season = ['seasonality']

print('# Demographic FIPS=', len(FIPS_demo), ', # Motality FIPS=', len(FIPS_mt), ', # Mobility FIPS=', len(FIPS_mb))
print('First date to be trained:', date_st, ', Final date to be trained:', date_ed)

if ON_CLUSTER:
    with open(PATH_CLUSTER, 'r') as f:
        classes = eval(f.read())

    cluster_size = max(classes.values()) # missing values are ignored
    FIPS_cluster = [set() for _ in range(cluster_size)]
    for fips, i in classes.items():
        if i!=cluster_size:
            FIPS_cluster[i].add(fips)
    for i in range(len(FIPS_cluster)):
        FIPS_cluster[i] = sorted(FIPS_cluster[i])
    """
    Generate training data
    in the order of demo-motal-mobi
    """
    for c in range(len(FIPS_cluster)):
        dataList = []
        for fips in FIPS_cluster[c]:
            data1 = demo[demo['fips']==fips][columns_demo].to_numpy()
            data1 = np.repeat(data1, len(date_win), axis=0)
            
            data2 = motality[(motality['fips']==fips) & (motality['date'].isin(date_win))][['date']+columns_mt]
            _ = [[dt, 0, 0] for dt in date_win if dt not in list(data2['date'])]
            data2 = data2.append(pd.DataFrame(_, columns=['date']+columns_mt))
            data2 = data2.sort_values(by=['date'])[columns_mt].to_numpy()
            
            data3 = mobility[(mobility['fips']==fips) & (mobility['date'].isin(date_win))][['date']+columns_mb]
            data3 = data3.sort_values(by=['date'])[columns_mb].to_numpy()

            dataList.append(np.hstack((data1, data2, data3)))
        np.save(f'{homedir}/JK/preprocessing/{md_now}/dataList_cls={c}.npy', np.asarray(dataList, dtype=np.float64))
        with open(f'{homedir}/JK/preprocessing/{md_now}/FIPS_cluster_cls={c}.txt', 'w') as f:
            print(FIPS_cluster[c], file=f)