'''
Plots
'''

import pandas as pd
import numpy as np
import plotly.express as px

def plot_yes_perc_error(X, y, pred):
    '''
    Create a scatterplot of the % yes and the error.
    '''

    ys = pd.DataFrame([X["municipality_dem"], X["yes_perc"], y, pred]).T
    ys.columns = ['municipality', 'yes_perc', 'y_test', 'y_pred']
    ys["error"] = np.absolute(ys["y_test"] - ys["y_pred"])
    
    fig = px.scatter(ys, x="yes_perc", y="error", color="y_test", 
                    hover_data=["municipality"], title="% Yes & Error Rate",
                    labels={
                        "yes_perc": "% Yes",
                        "error": "Error",
                        "y_test": "True Y",
                        "municipality": "Municipality"
                    })
    return fig