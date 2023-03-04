import pandas as pd
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