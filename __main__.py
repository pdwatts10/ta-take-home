import os
import re
import yaml
import requests
import pyarrow

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bokeh.plotting import figure, show
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIQuery:
    api: str
    year: int
    criterion_code: int
    query_limit: int
    skip_fetch: bool
    save_results: bool
    save_path: Optional[str] = None

    def __post_init__(self):
        if self.save_results and not self.save_path:
            raise ValueError("Must provide a valid file path if saving query results to dataframe")
        
@dataclass
class DataFilter:
    param: str
    operator: str
    threshold: float|int

@dataclass
class Scoring:
    param: str
    weight: float
    normalization: float|str

    def __post_init__(self):
        if isinstance(self.normalization,str) and self.normalization not in ('max','min','n/a'):
            raise ValueError("Invalid normalization criteria entered.\nNormalization criteria must be either a numeric value, 'max', or 'min'.\nIf no normalization is desired, enter 'n/a'")


def parse_config(config_path: str) -> tuple[APIQuery,dict[DataFilter],dict[Scoring],str,list[float],list[float]]:
    # load our config
    with open(config_path,'r') as fid:
        config = yaml.safe_load(fid)
    
    # parse query settings
    # mainly looking to get year, criterion code, and limits for results returned
    try:
        query = APIQuery(**config['api_query'])
    except Exception as e:
        print(f'Error parsing API query: {e}')

    # we'll want to filter our results for arrival dV, c3, etc., so parse those filters from the config
    filters = {}
    for key in config['selection_filters']:
        try:
            filters[config['selection_filters'][key]['param']] = DataFilter(**config['selection_filters'][key])
        except Exception as e:
            print(f'Error parsing filters: {e}')

    # we are provided primary and secondary metrics to evaluate, so let's assign scoring criteria for each metric
    scoring_criteria = {}
    for key in config['scoring_criteria']:
        try:
            scoring_criteria[config['scoring_criteria'][key]['param']] = Scoring(**config['scoring_criteria'][key])
        except Exception as e:
            print(f'Error parsing scoring criteria: {e}')
    
    # return class instantiations for our query config, data filters, and scoring criteria
    return query, filters, scoring_criteria, config['vehicle_data']['path'], config['vehicle_data']['dry_mass_targs'], config['vehicle_data']['isp_targs']

def parse_vehicle_data(path:str) -> pd.DataFrame:
    vehicle_data = pd.read_csv(path,header=0,index_col=0)
    
    c3_array = vehicle_data.columns
    c3_array = [float(re.findall(r'[0-9]{1,}$',ceng)[0]) for ceng in c3_array]
    vehicle_data = vehicle_data.rename(columns= {col:c3 for col,c3 in zip(vehicle_data.columns,c3_array)})

    return vehicle_data

def fetch_data(query: APIQuery) -> dict|None:
    # parse query options to construct api call
    url = f'{query.api}?lim={query.query_limit}&crit={query.criterion_code}&year={query.year}'

    # fetch data via api call, extract data if we get a good response
    # Raise exceptions for bad responses or other request exceptions
    try:
        response = requests.get(url) 
        response.raise_for_status()  
        data = response.json()       
    except requests.exceptions.HTTPError as e:  # if we don't get a return 200, raise an exception, return None for data
        print("HTTP error occurred:", e)
        data = None
    except requests.exceptions.RequestException as e:   # if get any other request exceptions, raise the exception, return None for data
        print("A request error occurred:", e)
        data = None
    
    return data

def process_data(query: APIQuery, data: dict|pd.DataFrame, filters: dict[DataFilter]|None,) -> pd.DataFrame:
    # web requests can be slow, time out, fail, etc., so if we had a chance to save the data we care about, let's pass it in directly
    # otherwise, construct a dataframe from the web request JSON data for data analysis later
    if isinstance(data,dict):
        df = pd.DataFrame(data['data'],columns=data['fields'])
        if query.save_results:
            df.to_parquet(query.save_path+'.parquet',engine='pyarrow')
            # df.to_csv(query.save_path+'.csv')
    else:
        df = data
        
    # if we passed an any queries, parse those and filter the data frame
    # tehcnically could skip this step if we saved a filtered dataframe already, but it's safer to assume we need to filter it every time
    if filters:
        df_filter = []
        for key in filters.keys():
            if df.dtypes[filters[key].param] == 'object':
                df[filters[key].param] = pd.to_numeric(df[filters[key].param])
            df_filter.append(' '.join([f'({filters[key].param}',f'{filters[key].operator}',f'{filters[key].threshold})']))

        df_filter = ' & '.join(df_filter)

        df = df.query(df_filter)
    
    # Save the dataframe for later use to skip web requests
    # exporting a csv or xlsx is also helpful to manually review outputs, debug problems with the query
    if query.save_results:
        df.to_parquet(query.save_path+'.parquet',engine='pyarrow')
        # df.to_csv(query.save_path+'.csv')

    return df

def plot_asteroid_data(asteroid_data:pd.DataFrame,scoring_criteria:dict[Scoring]):
    # when constructing visualizations, we'll focus on primary and secondary metrics
    # consume our scoring criteria to find which params we should plot
    plot_params = [scoring_criteria[key].param for key in scoring_criteria.keys()]
    
    # make sure all our metrics are numeric for plotting
    for param in plot_params:
        if asteroid_data.dtypes[param] == 'object':
            asteroid_data[param] = pd.to_numeric(asteroid_data[param])

    # seaborn's pair plot utility can plot every variable of interest against each other to quickly identify relationships between them
    sns.pairplot(data=asteroid_data,vars=plot_params)
    plt.show()

    pass

def select_target(asteroid_data:pd.DataFrame, scoring_criteria:dict[Scoring]) -> tuple[str,float]:
    # setup a collector for scores
    target_scores = {}

    # create a new data frame to hold our scores for each primary and secondary metric
    weighted_data = pd.DataFrame(index=asteroid_data['name'],columns=[scoring_criteria[key].param for key in scoring_criteria.keys()])

    # loop through all the params in our scoring criteria
    for key in scoring_criteria:
        # ensure we have numeric data to work with
        if asteroid_data.dtypes[scoring_criteria[key].param] == 'object':
            asteroid_data[scoring_criteria[key].param] = pd.to_numeric(asteroid_data[scoring_criteria[key].param])
        # to prevent skewing the results, normalize columns
        # normalization can be either the max or min value in a column or a user provided value
        match scoring_criteria[key].normalization:
            # normalize the column, multiply by the scoring weight, and assign to our new scoring data frame
            case 'max':
                normal_val = asteroid_data[scoring_criteria[key].param].max()
                scored = scoring_criteria[key].weight*asteroid_data[scoring_criteria[key].param].values/normal_val
                weighted_data.loc[:,scoring_criteria[key].param] = pd.Series(scored,index=weighted_data.index)
            case 'min':
                normal_val = asteroid_data[scoring_criteria[key].param].min()
                scored = scoring_criteria[key].weight*asteroid_data[scoring_criteria[key].param].values/normal_val
                weighted_data.loc[:,scoring_criteria[key].param] = pd.Series(scored,index=weighted_data.index)
            case float():
                normal_val = scoring_criteria[key].normalization
                scored = scoring_criteria[key].weight*asteroid_data[scoring_criteria[key].param].values/normal_val
                weighted_data.loc[:,scoring_criteria[key].param] = pd.Series(scored,index=weighted_data.index)
            case 'n/a':
                scored = scoring_criteria[key].weight*asteroid_data[scoring_criteria[key].param].values
                weighted_data.loc[:,scoring_criteria[key].param] = pd.Series(scored,index=weighted_data.index)

    # loop through each candidate target, sum its score, and collect
    for targ in weighted_data.index:
        target_scores[targ] = weighted_data.loc[targ,:].sum()

    # down select the target based on minimizing total score
    # minimized score represents lowest arrival dV, lowest c3, least orbit uncertainty, etc.
    final_target = min(target_scores,key=target_scores.get)

    # extract score for selected target
    score = target_scores[final_target]

    return final_target,score


def evaluate_total_mass(vinf_arr:float,dry_mass_range:list,isp_range:list,fig_height:float=500,fig_width:float=1000) -> pd.DataFrame:
    mass_mesh,isp_mesh = np.meshgrid(dry_mass_range,isp_range)
    perm = np.column_stack((mass_mesh.ravel(),isp_mesh.ravel()))
    

    spacecraft_mass = pd.DataFrame(columns=['total_mass','dry_mass','isp']) 
    for idx,(mass,isp) in enumerate(perm):
        spacecraft_mass.loc[idx,'total_mass'] = mass*np.exp(vinf_arr*1000./(isp*9.8))
        spacecraft_mass.loc[idx,'dry_mass'] = mass
        spacecraft_mass.loc[idx,'isp'] = isp

    sns.catplot(
        data=spacecraft_mass,x='dry_mass',y='total_mass',hue='isp',
        palette=['blue','red','green'],markers="o",linestyles="-",
        kind='point'
    )
    plt.grid(True)
    plt.xlabel('Dry Mass,kg')
    plt.ylabel('Spacecraft Mass,kg')
    plt.show()

    return spacecraft_mass

def interpolate_payload_mass(vehicle_data:pd.DataFrame,characteristic_energy: float,fig_height:float=500,fig_width:float=1000) -> dict[float]:

    # read in characteristic energy headings, ensure we strip out non-numeric info and parse into numeric array-like
    c3_array = vehicle_data.columns

    # Instantiate a figure for plotting payload capacities
    fig = figure(
        x_axis_label='Characteristic Energy, km^2/s^2',
        y_axis_label='Payload Capacity, kg',
        frame_height=fig_height,
        frame_width=fig_width
    )

    # Setup some color options for plotting
    colors = ['blue','red','green','black']

    # numpy can interpolate for us, so iterate through available vehicles, and return payload capacity for each at the input characteristic energy
    # in the same loop, we'll plot vehicle payload capacity as a function of characteristic energy
    # we'll also add the markers for the target payload mass for the input characteristic energy
    payload_capacity = {}
    for veh,color in zip(vehicle_data.index,colors):
        payload_array = vehicle_data.loc[veh,:]
        payload_capacity[veh] = float(np.interp(characteristic_energy,c3_array,payload_array))

        fig.line(
            x=c3_array,
            y=vehicle_data.loc[veh,:],
            line_width=2,
            legend_label=f'{veh} payload capacity',
            color=color
        )
        fig.scatter(
            x=characteristic_energy,
            y=payload_capacity[veh],
            size=10,
            legend_label=f'{veh} payload limit',
            color=color
        )
    
    # bokeh plots look great, and the interactive features work great when launched via web browser
    # however, legend placement is funky, and default layouts don't look great
    # bokeh exposes a method to reposition the legend though
    leg_obj = fig.legend[0]
    fig.add_layout(leg_obj,'right')
    show(fig)
    
    return payload_capacity

def plot_payload_capacity(vehicle_data:pd.DataFrame,characteristic_energy:float,payload_capacity:dict):

    # read in characteristic energy headings, ensure we strip out non-numeric info and parse into numeric array-like
    c3_array = vehicle_data.columns
    c3_array = [float(re.findall(r'[0-9]{1,}$',ceng)[0]) for ceng in c3_array]

    

    pass

def main(config_path:str):
    query, filters, scoring_criteria, vehicle_data_path, dry_mass_targs, isp_targs = parse_config(config_path)

    if query.skip_fetch and os.path.isfile(query.save_path):
        asteroid_data = pd.read_parquet(query.save_path)
    else:
        asteroid_data = fetch_data(query)
    asteroid_data = process_data(query,asteroid_data,filters)

    plot_asteroid_data(asteroid_data,scoring_criteria)
    target_asteroid,total_score = select_target(asteroid_data,scoring_criteria)

    vehicle_data = parse_vehicle_data(vehicle_data_path)

    target_vinf_arr, target_c3 = asteroid_data.loc[asteroid_data['name'] == target_asteroid,['vinf_arr','c3_dep']].values[0,:]

    spacecraft_mass = evaluate_total_mass(target_vinf_arr,dry_mass_targs,isp_targs)
    payload_capacity = interpolate_payload_mass(vehicle_data,target_c3)

    pass

if __name__ == "__main__":

    config_path = 'config.yaml'
    main(config_path)
    pass
