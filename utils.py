import pandas as pd
import matplotlib.pyplot as plt
import math
import re

def rename_columns_and_set_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the columns of a given dataframe to a more descriptive and readable format and
    set the type of the sch_date and op_date columns to datetime.

    Args:
        df (pandas.DataFrame): The input dataframe to rename the columns and set the datetime columns of.

    Returns:
        pandas.DataFrame: The input dataframe with renamed columns and datetime columns.
    """
    new_names = {
        'Fecha-I': 'sch_date',
        'Vlo-I': 'sch_num',
        'Ori-I': 'sch_origin',
        'Des-I': 'sch_dest',
        'Emp-I': 'sch_airline',
        'Fecha-O': 'op_date',
        'Vlo-O': 'op_num',
        'Ori-O': 'op_origin',
        'Des-O': 'op_dest',
        'Emp-O': 'op_airline',
        'DIA': 'day',
        'MES': 'month',
        'AÑO': 'year',
        'DIANOM': 'weekday',
        'TIPOVUELO': 'flight_type',
        'OPERA': 'op_airline_name',
        'SIGLAORI': 'origin_city',
        'SIGLADES': 'dest_city'
    }
    df = df.rename(columns=new_names)
    df['sch_date'] = pd.to_datetime(df['sch_date'])
    df['op_date'] = pd.to_datetime(df['op_date'])
    return df


def fix_flight_number(df: pd.DataFrame, column_list: list) -> pd.DataFrame:
    """
    Fix flight number column in a given dataframe by removing any non-numeric characters and converting to integers.

    Args:
        df (pandas.DataFrame): The input dataframe to fix the column in.
        column (str): The name of the column to fix.

    Returns:
        pandas.DataFrame: The input dataframe with the fixed column.
    """
    def convert_to_int(value):

        if pd.isna(value):
            return value
        
        try:
            float_value = float(value)
        except ValueError:
            float_value = None
        
        if isinstance(float_value, float):
            return math.trunc(float_value)
        
        return int(re.sub(r'\D', '', str(value)))
    
    for column in column_list:
        df[column] = df[column].apply(convert_to_int).astype('Int64')
    return df


def get_time_window(hour: int) -> str:
    """
    Returns the time window given an hour.

    Args:
        hour (int): The hour to convert to a time window.

    Returns:
        str: The corresponding time window.
    """
    if hour < 2:
        return '[0-2)'
    elif hour < 4:
        return '[2-4)'
    elif hour < 6:
        return '[4-6)'
    elif hour < 8:
        return '[6-8)'
    elif hour < 10:
        return '[8-10)'
    elif hour < 12:
        return '[10-12)'
    elif hour < 14:
        return '[12-14)'
    elif hour < 16:
        return '[14-16)'
    elif hour < 18:
        return '[16-18)'
    elif hour < 20:
        return '[18-20)'
    elif hour < 22:
        return '[20-22)'
    else:
        return '[22-24)'
    

def plot_delay_rate_by_group(df: pd.DataFrame, group_column: str, analysis_column: str, n_bars: int, legend_location: str, order: str ='Top') -> None:
    """
    Plots the delay rate by a given group column, considering the analysis column.

    Args:
        df (pd.DataFrame): The dataframe to be analyzed.
        group_column (str): The column name to group the dataframe.
        analysis_column (str): The column name to calculate the delay rate.
        n_bars (int): The number of bars to plot.
        legend_location (str): The location of the legend.

    Returns:
        None
    """

    # Delay rate by Group
    delay_rate_by_group = df.groupby(group_column)[analysis_column].mean().reset_index()

   
    if order == 'Top':
        top_n_group = delay_rate_by_group.nlargest(n_bars, analysis_column)
    elif order == 'Bottom':
        top_n_group = delay_rate_by_group.nsmallest(n_bars, analysis_column)

    delay_rate_by_group_pct = pd.DataFrame({group_column: top_n_group[group_column],
                                             'delay': top_n_group[analysis_column]*100,
                                             'on_time': (1-top_n_group[analysis_column])*100})

    ax = delay_rate_by_group_pct.plot(x=group_column, kind='bar', stacked=True, figsize=(10,6), 
                                      color=['red', 'green'], edgecolor='black', width=0.7)

    for i in ax.containers:
        ax.bar_label(i, labels=[f"{h:.1f}%" for h in i.datavalues], label_type='edge', fontsize=8)

    ax.set_title(f'Delay rate by {group_column} ({order} {n_bars})', fontsize=14)
    ax.legend(['Delay', 'On-time'], fontsize=12, loc=legend_location)
    plt.show()
