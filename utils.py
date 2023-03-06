import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime


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
        return '0-2'
    elif hour < 4:
        return '2-4'
    elif hour < 6:
        return '4-6'
    elif hour < 8:
        return '6-8'
    elif hour < 10:
        return '8-10'
    elif hour < 12:
        return '10-12'
    elif hour < 14:
        return '12-14'
    elif hour < 16:
        return '14-16'
    elif hour < 18:
        return '16-18'
    elif hour < 20:
        return '18-20'
    elif hour < 22:
        return '20-22'
    else:
        return '22-24'
    

def plot_delay_rate_by_group(df: pd.DataFrame, group_column: str, analysis_column: str, n_bars: int, legend_location: str, order: str ='Top') -> None:
    """
    Plots the delay rate by a given group column, considering the analysis column.

    Args:
        df (pandas.DataFrame): The dataframe to be analyzed.
        group_column (str): The column name to group the dataframe.
        analysis_column (str): The column name to calculate the delay rate.
        n_bars (int): The number of bars to plot.
        legend_location (str): The location of the legend.

    Returns:
        None: Displays the plot on the screen.
    """
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


def one_hot_encode(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Applies one-hot encoding to the specified categorical columns of a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The pandas DataFrame to be encoded.
        cols (list): List of column names to be encoded.

    Returns:
        pandas.DataFrame: The encoded pandas DataFrame.
    """

    for col in cols:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, axis=1)
        df = df.join(one_hot)
    return df


def plot_corr_matrix(df, figsize=(12, 10), text_annotations=True, xticklabels=True, yticklabels=True):
    """
    Plots a correlation matrix based on the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing numerical data.
        figsize (tuple), optional (default=(12, 10)): The size of the plot in inches.
        text_annotations (bool), optional (default=True): Whether to show the numerical correlation values as text annotations.
        xticklabels (bool), optional (default=True): Whether to show x-axis tick labels (variable names).
        yticklabels (bool), optional (default=True): Whether to show y-axis tick labels (variable names).

    Returns:
        None: Displays the plot on the screen.
    """
    corr = df.corr()
    corr = np.tril(corr)  # asignar cero a los elementos por encima de la diagonal principal

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=12)

    if xticklabels:
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticks([])

    if yticklabels:
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_yticklabels(corr.columns, fontsize=10)
    else:
        ax.set_yticks([])

    if text_annotations:
        for i in range(len(corr.columns)):
            for j in range(i): # rango modificado para mostrar solo diagonal inferior
                text = ax.text(j, i, round(corr[i, j], 2), ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Correlation Matrix", fontsize=16)
    fig.tight_layout()

    plt.show()


def find_highly_correlated_vars(df, corr_threshold=0.95):
    """
     Finds variables in a DataFrame with correlation above a certain threshold.

    Args:
        df (pandas.DataFrame): The input DataFrame containing numerical data.
        corr_threshold (float), optional (default=0.95): The minimum absolute correlation value to consider a pair of variables highly correlated.

    Returns:
        list: A list of variable names that have high correlation with at least one other variable in the DataFrame.
    """
    corr_matrix = df.corr()
    corr_pairs = corr_matrix.stack().reset_index()
    corr_pairs.columns = ['var1', 'var2', 'corr']
    corr_pairs = corr_pairs.loc[corr_pairs['var1'] < corr_pairs['var2']]
    sorted_pairs = corr_pairs[abs(corr_pairs['corr']) > corr_threshold].sort_values(by='corr', ascending=False)
    vars_to_drop = []
    for i, row in sorted_pairs.iterrows():
        if row['var1'].startswith("sch_airline_name_"):
            var_0 = row['var1']
            row['var1'] = row['var2']
            row['var2'] = var_0
        vars_to_drop.append(row['var2'])
        print(f"{row['var1']} and {row['var2']}: {row['corr']:.2f}")
    return vars_to_drop


def get_correlations_with_variable(df, variable_name, lower_threshold=0, upper_threshold=1):
    """
    Returns a DataFrame with the correlations of all variables with a specific variable, sorted by absolute value.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        variable_name (str): The name of the variable to which correlations are being computed.
        lower_threshold (float), optional (default=0): The lower threshold to filter the correlations by absolute value.
        upper_threshold (float), optional (default=1): The upper threshold to filter the correlations by absolute value.

    Returns:
        pandas.DataFrame: A DataFrame with the absolute value and sign of the correlations of all variables with the specific variable, sorted by absolute value.
    """
    if lower_threshold < 0 or lower_threshold > 1 or upper_threshold < 0 or upper_threshold > 1:
        raise ValueError("Threshold values must be between 0 and 1.")

    corr_matrix = df.corr()[variable_name]
    corr_matrix = corr_matrix.drop(index=variable_name)
    corr_matrix = corr_matrix[(abs(corr_matrix) >= lower_threshold) & (abs(corr_matrix) <= upper_threshold)]
    corr_matrix = corr_matrix.sort_values(ascending=False, key=lambda x: abs(x))
    corr_matrix.name = 'corr'
    corr_matrix.index.name = 'var'
    corr_matrix = corr_matrix.reset_index()
    corr_matrix['corr_sign'] = np.sign(corr_matrix['corr'])
    corr_matrix['corr_abs'] = abs(corr_matrix['corr'])
    return corr_matrix[['var', 'corr_sign', 'corr_abs']]


def split_data(df, target, test_size=0.2, val_size=0.25, random_state=42):
    """
    Splits data into train, validation, and test sets.

    Args:
        df (pandas.DataFrame): Dataframe containing the data to be split.
        target (str): Name of the target variable column.
        test_size (float), optional: Size of the test set. Defaults to 0.2.
        val_size (float), optional: Size of the validation set. Defaults to 0.25.
        random_state (int), optional: Random state to use for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the train, validation, and test sets as pandas DataFrames.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=test_size, stratify=df[target], random_state=random_state)

    # Split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models(X_train, y_train, X_val, y_val, models, params, cv=5, scoring='f1'):
    """
    Trains multiple machine learning models using GridSearchCV and returns the top models ranked by selected score. 

    Args:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_val (array-like): Validation data features.
        y_val (array-like): Validation data labels.
        models (dict): Dictionary containing the models to train.
        params (dict): Dictionary containing the hyperparameters for each model.
        cv (int), optional: Number of cross-validation folds. Defaults to 5.
        scoring (str), optional: Scoring metric to use. Defaults to 'f1'.

    Returns:
        tuple: A tuple containing a list of the trained top 5 models and a list of the top 5 models with their scores.

    """
    print(f" - Started at: {str(datetime.now())}")
    best_models = {}
    trained_models = []
    for name, estimator in models.items():
        print(f"Training {name}...")
        clf = GridSearchCV(estimator, params[name], cv=cv, scoring=scoring)
        clf.fit(X_train, y_train)
        print(f" - Best params: {clf.best_params_}")
        print(f" - Best {scoring}: {clf.best_score_:.4f}")
        y_pred_train = clf.predict(X_train)
        f1_train = f1_score(y_train, y_pred_train)
        print(f" - {scoring} on train set: {f1_train:.4f}")
        y_pred_val = clf.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val)
        print(f" - {scoring} on validation set: {f1_val:.4f}")
        best_models[name] = {'model': clf.best_estimator_, 'f1_score': clf.best_score_, 'f1_train': f1_train, 'f1_val': f1_val}
        trained_models.append(clf.best_estimator_)
        print(f" - Ended at: {str(datetime.now())}")
    
    top_models = sorted(best_models.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    top_models_list = [model[1]['model'] for model in top_models]
    return top_models_list, top_models


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
    ax.set_ylabel('Actual outputs', fontsize=12, color='black')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, color='black')
    ax.xaxis.set(ticks=range(len(np.unique(y_true))), ticklabels=np.unique(y_true))
    ax.yaxis.set(ticks=range(len(np.unique(y_true))), ticklabels=np.unique(y_true))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    for i in range(len(np.unique(y_true))):
        for j in range(len(np.unique(y_true))):
            ax.text(j, i, format(cm[i, j], 'd'), color='red', fontsize=15, ha='center', va='center')
    plt.show()
    
    
def evaluate_models(kfolded_models, X_test, y_test, threshold=0.5):
    """
    Evaluates multiple machine learning models using classification metrics and confusion matrix.

    Args:
        kfolded_models (list): List containing the trained kfolded models.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.

    Returns:
        None.

    """
    for model in kfolded_models:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)
        print(f"Model: {type(model).__name__}")
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, type(model).__name__)