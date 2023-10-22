import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors
from pathlib import Path


def plot_time_series(df_to_visual: pd.DataFrame,
                     y_axis_map_names: list[str],
                     y_plots: list[list[str]],
                     y_axis_names: list[str],
                     y_places: list[float],
                     plot_position: list[float],
                     titles: dict[str],
                     savepath: Path,
                     colors: matplotlib.colors.LinearSegmentedColormap
                     ) -> None:
    """_summary_

    Args:
        df_to_visual (pd.DataFrame): _description_
        y_axis_map_names (list[str]): _description_
        y_plots (list[list[str]]): _description_
        y_axis_names (list[str]): _description_
        y_places (list[float]): _description_
        plot_position (list[float]): _description_
        titles (dict[str]): _description_
        savepath (str): _description_
        colors (matplotlib.colors.LinearSegmentedColormap): _description_
    """
    
    fig = go.Figure()

    for y_axis_name, cols in zip(y_axis_map_names, y_plots):
        for col in cols:
            fig.add_trace(go.Scatter(
                x=df_to_visual[col].dropna().index,
                y=df_to_visual[col].dropna(),
                name=col,
                yaxis=y_axis_name,
            ))

    yaxis_dict = {}
    for i, (y_axis_name, y_pos) in enumerate(zip(y_axis_names, y_places)):
        color = matplotlib.colors.to_hex(colors(i), keep_alpha=False)
        name = f'yaxis{i+1}'
        add_params = dict(
            anchor="free",  # specifying x - axis has to be the fixed
            overlaying="y",  # specifyinfg y - axis has to be separated
        )
        if i == 0:
            name = "yaxis"
            add_params = {}
        yaxis_dict[name] = dict(
            title=y_axis_name,
            titlefont=dict(color=color),
            tickfont=dict(color=color),
            side="left" if y_pos <= plot_position[0] else "right",
            position=y_pos,
            **add_params,
        )

    # Create axis objects
    fig.update_layout(
        # split the x-axis to fraction of plots in
        # proportions
        xaxis=dict(
            domain=plot_position
        ),
        # pass the y-axis title, titlefont, color
        # and tickfont as a dictionary and store
        # it an variable yaxis
        **yaxis_dict
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=14, label="14d", step="day", stepmode="todate"),
                dict(step="all")
            ])
        )
    )

    # Update layout of the plot namely title_text, width
    # and place it in the center using title_x parameter
    # as shown
    fig.update_layout(
        title_text=titles["title_text"],
        xaxis_title=titles["xaxis_title"],
        hovermode='x unified',
        height=1000,
        width=1600,
        title_x=0.5
    )
    # fig.update_traces(mode="markers+lines", hovertemplate=None)
    # fig.update_layout(hovermode="x")

    fig.write_html(savepath)
