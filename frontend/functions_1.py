from functions_dashboard import *
import joblib
import numpy as np


def kdeplot_in_common(X_split_valid, y_split_valid, feature, bw_method=0.4):
    """KDE plot of a quantitative feature. Common to all clients.
    Args :
    - feature (string).
    Returns :
    - matplotlib figure.
    """
    # Extraction of the feature's data
    df = pd.DataFrame({
        feature: X_split_valid[feature],
        'y_true': y_split_valid
    })
    ser_true0 = df.loc[df['y_true'] == 0, feature]
    ser_true1 = df.loc[df['y_true'] == 1, feature]
    xmin = df[feature].min()
    xmax = df[feature].max()
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4, dpi=100)
    ser_true0.plot(kind='kde',
                   c='g',
                   label='Non-defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    ser_true1.plot(kind='kde',
                   c='r',
                   label='Defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    fig.suptitle(
        f'Observed distribution of {feature} based on clients true class',
        y=0.92)
    plt.legend()
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.xlim(xmin, xmax)
    return fig


def kdeplot(one_client_pandas, client_id, feature):
    """Plots a KDE of the quantitative feature.
    Args :
    - feature (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    if feature in [
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'AMT_ANNUITY'
    ]:
        figure = joblib.load('./resources/figure_kde_distribution_' + feature +
                             '_for_datascientist.joblib')
    else:
        figure = kdeplot_in_common(feature)
    y_max = plt.ylim()[1]
    x_client = one_client_pandas[feature].iloc[0]
    if str(x_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        plt.annotate(text=f" Client {client_id}\n  data not available",
                     xy=(x_center, 0),
                     xytext=(x_center, y_max * 0.8))
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=2)
        plt.annotate(text=f" Client {client_id}\n  {round(x_client, 3)}",
                     xy=(x_client, y_max * 0.8))
    st.pyplot(figure)
    st.caption(feature + ": " + feature_description(feature))


def barplot_in_common(X_split_valid, y_split_valid, dict_categorical_features, feature):
    """Horizontal Barplot of a qualitative feature. Common to all clients.
    Args :
    - feature (string).
    Returns :
    - matplotlib figure.
    """
    # Extraction of the feature's data
    df_feature = pd.DataFrame({
        feature: X_split_valid[feature],
        'y_true': y_split_valid
    })
    # Observed probability of default for each value of the feature
    proba_for_each_value = []
    cardinality = len(
        dict_categorical_features[feature]) if feature != 'CODE_GENDER' else 2
    for index in range(
            cardinality):  # on parcourt toutes les modalités de la feature
        df_feature_modalite = df_feature[df_feature[feature] == index]
        proba_default = df_feature_modalite['y_true'].sum() / len(
            df_feature_modalite)
        proba_for_each_value.append(proba_default)
    df_modalites = pd.DataFrame()
    df_modalites['modalites'] = dict_categorical_features[
        feature] if feature != 'CODE_GENDER' else ['Female', 'Male']
    df_modalites['probas'] = proba_for_each_value
    df_modalites.sort_values(by='probas', inplace=True)
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.ylim(-0.6, cardinality - 0.4)
    plt.barh(y=range(cardinality), width=df_modalites['probas'], color='r')
    plt.barh(
        y=range(cardinality),
        left=df_modalites['probas'],
        width=(1 - df_modalites['probas']),
        color='limegreen',
    )
    plt.xlabel('Observed probability of default')
    plt.ylabel(feature)
    fig.suptitle(
        f'Observed probability of default as a function of {feature} based on clients true class',
        y=0.92)
    size = 6 if cardinality > 30 else None
    plt.yticks(ticks=range(cardinality),
               labels=df_modalites['modalites'],
               size=size)
    return fig


def barplot(one_client_pandas, optimum_threshold, client_id, dict_categorical_features, feature):
    """Barplot of a qualitative feature.
    Args :
    - feature (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    if feature in ['ORGANIZATION_TYPE', 'CODE_GENDER']:
        figure = joblib.load('./resources/figure_barplot_' + feature +
                             '_for_datascientist.joblib')
    else:
        figure = barplot_in_common(feature)
    x_client = one_client_pandas[feature].iloc[0]
    plt.axvline(x=optimum_threshold,
                ymin=-1e10,
                ymax=1e10,
                c='darkorange',
                ls='dashed',
                lw=1)  # line for the optimum_threshold
    plt.text(
        s=
        f" Client {client_id}: {dict_categorical_features[feature][x_client]} ",
        x=0.5,
        y=plt.ylim()[1] * 0.3)
    st.pyplot(figure)
    st.caption(feature + ": " + feature_description(feature))


def contourplot_in_common(X_split_valid, y_split_valid, feature1, feature2):
    """Contour plot for the observed probability of default as a function of 2 features. Common to all clients.
    Args :
    - feature1 (string).
    - feature2 (string).
    Returns :
    - matplotlib figure.
    """
    target_mesh_size = 500  # target population for each mesh

    # Preparation of data
    df = pd.DataFrame({
        feature1: X_split_valid[feature1],
        feature2: X_split_valid[feature2],
        'y_true': y_split_valid
    })
    df = df.dropna().copy()
    n_values = len(df)
    n_bins = int(np.ceil(np.sqrt(n_values / target_mesh_size)))
    bin_size = int(np.floor(n_values / n_bins))
    index_bin_start = sorted([bin_size * n for n in range(n_bins)])
    ser1 = df[feature1].sort_values().copy()
    ser2 = df[feature2].sort_values().copy()

    # Filling the grid
    grid_proba_default = np.full((n_bins, n_bins), -1.0)
    ser_true0 = (df['y_true'] == 0)
    ser_true1 = (df['y_true'] == 1)
    for i1, ind1 in enumerate(index_bin_start):
        for i2, ind2 in enumerate(index_bin_start):
            ser_inside_this_mesh = (df[feature1] >= ser1.iloc[ind1]) & (df[feature2] >= ser2.iloc[ind2]) \
                                   & (df[feature1] <= ser1.iloc[ind1 + bin_size - 1]) & (
                                               df[feature2] <= ser2.iloc[ind2 + bin_size - 1])
            # sum of clients true0 inside this square bin
            sum_0 = (ser_inside_this_mesh & ser_true0).sum()
            sum_1 = (ser_inside_this_mesh & ser_true1).sum()
            sum_ = sum_0 + sum_1
            if sum_ == 0:
                proba_default = 1
            else:
                proba_default = sum_1 / sum_
            grid_proba_default[i2, i1] = proba_default

    # X, Y of the grid
    X = [ser1.iloc[i + int(bin_size / 2)] for i in index_bin_start]
    Y = [ser2.iloc[i + int(bin_size / 2)] for i in index_bin_start]

    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.contourf(X, Y, grid_proba_default, cmap='Reds')
    plt.colorbar(shrink=0.8)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    fig.suptitle(
        f'Observed probability of default as a function of {feature1} and {feature2}',
        y=0.92)
    return fig


def contourplot(one_client_pandas, client_id, feature1, feature2):
    """Contour plot for the observed probability of default as a function of 2 features.
    Args :
    - feature1 (string).
    - feature2 (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    figure = contourplot_in_common(feature1, feature2)
    x_client = one_client_pandas[feature1].iloc[0]
    y_client = one_client_pandas[feature2].iloc[0]
    if str(x_client) == "nan" or str(y_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        y_center = (plt.ylim()[1] + plt.ylim()[0]) / 2
        plt.text(s=f" Client {client_id}\n  data not available",
                 x=x_center,
                 y=y_center)
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=1)
        plt.axhline(y=y_client,
                    xmin=-1e10,
                    xmax=1e10,
                    c='k',
                    ls='dashed',
                    lw=1)
        # if I want to interpolate data : https://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines
    st.pyplot(figure)
    st.caption(feature1 + ": " + feature_description(feature1))
    st.caption(feature2 + ": " + feature_description(feature2))





def plot_selector(list_categorical_features, feature, dashboard='Advanced Dashboard'):
    """Chooses between a KDE plot (for quantitative features) and a bar plot (for qualitative features)
    Args :
    - feature (string).
    - dashboard (string) : 'Advanced Dashboard' or 'Basic Dashboard'.
    Returns :
    - matplotlib plot via st.pyplot of the called function.
    """
    if feature in list_categorical_features:
        barplot(feature)
    else:
        if dashboard == 'Advanced Dashboard':
            kdeplot(feature)
        else:
            lineplot(feature)


def display_EDA(file_name):
    """Loads a HTML file generated by dataprep and displays it.
    Args:
    - file_name: HTML file compressed into a joblib file.
    Returns:
    - display of the HTML file via st.components.v1.html().
    """
    report = joblib.load('./resources/eda/' + file_name + '.joblib')
    st.components.v1.html(report, width=1200, height=800, scrolling=True)



#####################################################################################################################
