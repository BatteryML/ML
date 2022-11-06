import pandas as pd
import numpy as np
from scipy.io.arff import loadarff 
from scipy.stats import kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, homogeneity_score, mean_squared_error, f1_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numbers
import seaborn as sns
import time
plt.rcParams["figure.dpi"] = 130

# read data
def load_data1():
    data1_name = 'Telescope'
    raw_data1 = loadarff('MagicTelescope.arff')
    data1 = pd.DataFrame(raw_data1[0])
    data1 = data1.drop(columns=['ID'])
    data1 = data1.rename(columns=lambda x: x.replace(':', ''))
    data1['class'] = data1['class'].str.decode('utf-8')
    data1 = pd.concat([
        data1[data1['class']=='g'].sample(510, random_state=0),
        data1[data1['class']=='h'].sample(490, random_state=0)])
    data1.index = range(len(data1))
    X1=data1.drop(columns=['class'])
    y1=data1['class']
    min_max_scaler = MinMaxScaler()
    X1_scaled = min_max_scaler.fit_transform(X1)
    X1_scaled = pd.DataFrame(X1_scaled, columns=X1.columns)

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.3, random_state=0, stratify=y1)
    X1_train_train, X1_val, y1_train_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.2, random_state=0, stratify=y1_train)
    return data1_name, X1_scaled, y1, X1_train, X1_test, y1_train, y1_test, X1_train_train, X1_val, y1_train_train, y1_val

def load_data2():
    data2_name = 'Wine'
    raw_data2 = loadarff('wine-quality-red.arff')
    data2 = pd.DataFrame(raw_data2[0])
    data2['class'] = data2['class'].str.decode('utf-8')
    data2 = pd.concat([
        data2[data2['class']=='5'].sample(400, random_state=0),
        data2[data2['class']=='6'].sample(400, random_state=0),
        data2[data2['class']=='7']])
    data2.index = range(len(data2))
    X2=data2.drop(columns=['class'])
    y2=data2['class']
    min_max_scaler = MinMaxScaler()
    X2_scaled = min_max_scaler.fit_transform(X2)
    X2_scaled = pd.DataFrame(X2_scaled, columns=X2.columns)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.3, random_state=0, stratify=y2)
    X2_train_train, X2_val, y2_train_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.2, random_state=0, stratify=y2_train)
    return data2_name, X2_scaled, y2, X2_train, X2_test, y2_train, y2_test, X2_train_train, X2_val, y2_train_train, y2_val

def select_K_kmeans(X_train, data_name, max_k=11, figsize=(5,2)):
    train_SSE_list = []
    train_silhouette_score_list = []
    k_list = list(range(1, max_k))
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train)
        train_SSE_list.append(-kmeans.score(X_train))
        if k >= 2:
            train_silhouette_score_list.append(silhouette_score(X_train, kmeans.predict(X_train)))
    plt.figure(figsize=figsize)
    plt.plot(k_list, train_SSE_list, '--o', label='Train')
    plt.xticks(k_list)
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title(f'Elbow Curve Method -- {data_name}')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=figsize)
    plt.plot(k_list[1:], train_silhouette_score_list, '--o', label='Train')
    plt.xticks(k_list[1:])
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score Method -- {data_name}')
    plt.legend()
    plt.show()
    
def select_K_em(X_train, data_name, max_k=11, figsize=(5,2)):
    train_logl_list = []
    train_silhouette_score_list = []
    k_list = list(range(1, max_k))
    for k in k_list:
        em = GaussianMixture(n_components=k, random_state=0).fit(X_train)
        train_logl_list.append(em.score(X_train))
        if k >= 2:
            train_silhouette_score_list.append(silhouette_score(X_train, em.predict(X_train)))
    
    plt.figure(figsize=figsize)
    plt.plot(k_list, train_logl_list, '--o', label='Train')
    plt.xticks(k_list)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Log-likelihood')
    plt.title(f'Elbow Curve Method -- {data_name}')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=figsize)
    plt.plot(k_list[1:], train_silhouette_score_list, '--o', label='Train')
    plt.xticks(k_list[1:])
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score Method -- {data_name}')
    plt.legend()
    plt.show()
    
def tune_hyper(X_train, y_train, classifier, param_grid):
    skfolds = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    search_f1 = GridSearchCV(classifier, param_grid, cv=skfolds, scoring='f1_weighted', return_train_score=True, n_jobs=-1)
    search_f1.fit(X_train.values, y_train.values)
    best_classifier = search_f1.best_estimator_
    best_f1 = search_f1.best_score_
    best_params = search_f1.best_params_
    cv_result = pd.DataFrame(search_f1.cv_results_)
    print(f'Best classifier is {best_classifier} with f1 score {best_f1}')
    return best_params, cv_result

def plot_validation_curve(hyper, params, cv_result, model_type, data_name, hyper_name=None, log_x=False, rot_x=False):
    if hyper_name is None:
        hyper_name = hyper
    cv_result_hyper = cv_result.copy()
    change_ticks = False
    for param in params:
        if param != hyper:
            cv_result_hyper = cv_result_hyper[cv_result_hyper[f'param_{param}']==params[param]]
    hyper_list = cv_result_hyper[f'param_{hyper}']
    f1_train_list = cv_result_hyper['mean_train_score']
    f1_val_list = cv_result_hyper['mean_test_score']
    if not isinstance(list(hyper_list)[0], numbers.Number):
        change_ticks = True
        tick_label_list = hyper_list
        hyper_list = list(range(len(hyper_list)))
    plt.figure(figsize=(5,3))
    plt.plot(hyper_list, f1_train_list, '-o', label='train')
    plt.plot(hyper_list, f1_val_list, '-o', label='validation')
    if log_x:
        plt.xscale('log')
    if change_ticks:
        rotation = 0
        if rot_x == True:
            rotation = 45
        plt.xticks(hyper_list, tick_label_list, rotation=rotation)
    plt.xlabel(f'{hyper_name}')
    plt.ylabel('Performance (F1 Score)')
    plt.legend()
    plt.title(f'{model_type} Validation Curve  - {data_name}')
    plt.show()
    
def get_f1_avg(X_train, y_train, test_size, classifier):
    f1_train_train_list = []
    f1_val_list = []
    for random_state in range(10):
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train)
        # test_idx = int(test_size*len(X_train))
        # X_train_train = X_train.iloc[test_idx:,:]
        # y_train_train = y_train.iloc[test_idx:]
        # X_val = X_train.iloc[:test_idx,:]
        # y_val = y_train.iloc[:test_idx]
        classifier.fit(X_train_train, y_train_train)
        pred_train_train = classifier.predict(X_train_train)
        f1_train_train=f1_score(y_train_train, pred_train_train, average="weighted")
        pred_val = classifier.predict(X_val)
        f1_val=f1_score(y_val, pred_val, average="weighted")
        f1_train_train_list.append(f1_train_train)
        f1_val_list.append(f1_val)
    return np.mean(f1_train_train_list), np.mean(f1_val_list)

def get_f1_score_vs_train_pct(X_train, y_train, classifier):
    f1_train_train_list=[]
    f1_val_list=[]
    thresh_pct_list=[]
    for i in range(1,10):
        pct=i/10
        thresh_pct_list.append(i*10)
        f1_train_train, f1_val = get_f1_avg(X_train, y_train, test_size=1-pct, classifier=classifier)
        f1_train_train_list.append(f1_train_train)
        f1_val_list.append(f1_val)
    summary = pd.DataFrame({
        'f1_train': f1_train_train_list,
        'f1_val': f1_val_list,
        'train_pct': thresh_pct_list
    })   
    display(summary)
    return (f1_train_train_list, f1_val_list, thresh_pct_list)

def plot_learning_curve(f1, model_type, data_name):
    plt.figure(figsize=(4,2.5))
    plt.plot(f1[2], f1[0], '-o', label='train')
    plt.plot(f1[2], f1[1], '-o', label='validation')
    plt.xticks(f1[2])
    plt.xlabel('Training Size (%)')
    plt.ylabel('Performance (F1 Score)')
    plt.title(f'{model_type} Learning Curve - {data_name}')
    plt.legend()
    plt.show()
    
def get_loss_per_iter(X_train_train, y_train_train, X_val, y_val, params, max_iter=100, iter_step=5):
    loss_train_train_list = []
    loss_val_list = []
    iter_list = []
    classifier = MLPClassifier(**params, warm_start=True, max_iter=iter_step, random_state=0)
    for i in range(0,max_iter,iter_step):
        classifier.fit(X_train_train.values, y_train_train.values)
        pred_train_train = classifier.predict_proba(X_train_train.values)
        loss_train_train = log_loss(y_train_train, pred_train_train)
        loss_train_train_list.append(loss_train_train)
        pred_val = classifier.predict_proba(X_val.values)
        loss_val = log_loss(y_val, pred_val)
        loss_val_list.append(loss_val)
        iter_list.append(i)
    return loss_train_train_list, loss_val_list, iter_list

def plot_learning_curve_loss_iter(loss, model_type, data_name):
    plt.figure(figsize=(4,2.5))
    plt.plot(loss[2], loss[0], '-o', label='train')
    plt.plot(loss[2], loss[1], '-o', label='validation')
    tick_step = (loss[2][1] - loss[2][0]) * 2
    plt.xticks(range(min(loss[2]), max(loss[2])+tick_step, tick_step))
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.title(f'{model_type} Learning Curve Loss vs. Iteration - {data_name}')
    plt.legend()
    plt.show()
    
def model_performance(X_train, y_train, X_test, y_test, classifier, model_type, data_name):
    start_t = time.time()
    classifier.fit(X_train, y_train)
    train_t = round(time.time() - start_t, 4)
    start_t = time.time()
    for i in range(100):
        pred_test = classifier.predict(X_test)
    pred_t = round((time.time() - start_t)/100, 4)
    pred_train = classifier.predict(X_train)
    print('Test performance:')
    print(classification_report(y_test, pred_test))
    cm = confusion_matrix(y_test, pred_test, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    fig = plt.figure(figsize=(5,3))
    disp.plot(ax=fig.gca())
    plt.title(f'{model_type} - {data_name} - Test')
    plt.show()
    print('Train performance:')
    print(classification_report(y_train, pred_train))
    cm = confusion_matrix(y_train, pred_train, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    fig = plt.figure(figsize=(5,3))
    disp.plot(ax=fig.gca())
    plt.title(f'{model_type} - {data_name} - Train')
    plt.show()
    print(f'Train time is {train_t} seconds.')
    print(f'Prediction time is {pred_t} seconds.')
    return train_t, pred_t
    
def plot_pairwise(X_train, cluster, data_name, palette='tab10'):
    df = X_train.copy()
    df['cluster'] = cluster
    sns.pairplot(df, hue='cluster', palette=palette)
    plt.title(f'Pairwise Plot -- {data_name}')

    # fig, ax = plt.subplots(figsize=(8,8))
    # axes = pd.plotting.scatter_matrix(X_train, diagonal='hist', marker='.', c=cluster, 
    #                               range_padding=0.2, ax=ax)
    # for subax in axes.flatten():
    #     subax.xaxis.label.set_rotation(90)
    #     subax.yaxis.label.set_rotation(0)
    #     subax.yaxis.label.set_ha('right')
    # labels = sorted(np.unique(cluster))
    # plt.legend(ax.legend_elements()[0], labels,loc=(1.02,0))
    # plt.show()
    
def plot_homogeneity(y_train, cluster, data_name):
    df = pd.DataFrame({'label': y_train, 'cluster': cluster}).groupby(['label', 'cluster']).size()
    df.rename('count').reset_index().pivot(index='cluster', columns='label', values='count').plot.bar(
        stacked=True,
        rot=0,
        ylabel='count',
        figsize=(3.5,2.5),
        title=f'Homogeneity -- {data_name}')
    plt.show()
    print(f'Homogeneity score is {homogeneity_score(y_train, cluster): .6f}')
    
def plot_PCA(exp_var_pca, cum_exp_var_pca, data_name):
    x = range(1, len(exp_var_pca)+1)
    plt.figure(figsize=(4,3))
    plt.bar(x, exp_var_pca, align='center', label='Individual')
    plt.step(x, cum_exp_var_pca, where='mid', color='r', label='Cumulative')
    xlim = plt.gca().get_xlim()
    plt.hlines(y=0.9, xmin=xlim[0], xmax=xlim[1], colors=['grey'], linestyles='--', alpha=0.5)
    plt.xticks(x)
    plt.yticks(np.linspace(0,1,11))
    plt.xlim(xlim)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title(f'PCA Explained Variance -- {data_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ICA(data_ica, data_name):
    # kurtosis
    kurto_ica = np.abs(kurtosis(data_ica, fisher=True)) # fisher: normal --> 0
    component_list = kurto_ica.argsort()[::-1] + 1
    kurto_ica = sorted(kurto_ica, reverse=True)
    print(kurto_ica)
    # # accumulative
    # kurto_sum_ica = np.cumsum(kurto_ica)
    # kurto_mean_ica = kurto_sum_ica/(np.arange(0,data_ica.shape[1])+1)
    x = range(1, len(kurto_ica)+1)
    plt.figure(figsize=(4,2.5))
    plt.plot(x, kurto_ica, '--o')
    plt.xticks(x, component_list)
    plt.xlabel('Component Index')
    plt.ylabel('Component Kurtosis')
    plt.title(f'ICA Kurtosis -- {data_name}')
    plt.show()
    
# def mean_kurto_ica(X_train):
#     N_list = list(range(1, X_train.shape[1]+1))
#     mean_kurto = []
#     for N in N_list:
#         ica = FastICA(n_components=N, random_state=0, whiten='arbitrary-variance')
#         data_ica = ica.fit_transform(X_train)
#         mean_kurto.append(round(np.mean(np.abs(kurtosis(data_ica,  fisher=True))), 6))
#     return N_list, mean_kurto

# def plot_ICA(X_train, data_name):
#     N_list, mean_kurto = mean_kurto_ica(X_train)
#     print(mean_kurto)
#     plt.figure(figsize=(4,2.5))
#     plt.plot(N_list, mean_kurto, '--o')
#     plt.xticks(N_list)
#     plt.xlabel('Component Number')
#     plt.ylabel('Average Kurtosis')
#     plt.title(f'ICA Average Kurtosis -- {data_name}')
#     plt.show()
        
def recon_error_rp(X_train):
    N_list = list(range(1, X_train.shape[1]+1))
    recon_error = []
    for N in N_list:
        rp = SparseRandomProjection(n_components=N, random_state=0)
        data_rp = rp.fit_transform(X_train)
        inverse_rp = rp.inverse_transform(data_rp)
        recon_error.append(mean_squared_error(X_train, inverse_rp))
    return N_list, recon_error

def recon_error_pca(X_train):
    N_list = list(range(1, X_train.shape[1]+1))
    recon_error = []
    for N in N_list:
        pca = PCA(n_components=N, random_state=0)
        data_pca = pca.fit_transform(X_train)
        inverse_pca = pca.inverse_transform(data_pca)
        recon_error.append(mean_squared_error(X_train, inverse_pca))
    return N_list, recon_error

def plot_RP(X_train, data_name):
    N_list, recon_error = recon_error_rp(X_train)
    plt.figure(figsize=(4,2.5))
    plt.plot(N_list, recon_error, '--o')
    plt.xticks(N_list)
    plt.xlabel('Component Number')
    plt.ylabel('Reconstruction Error')
    plt.title(f'RP Reconstruction Error -- {data_name}')
    plt.show()
    
def compare_RP_and_PCA(X_train, data_name):
    N_list, recon_error1 = recon_error_rp(X_train)
    N_list, recon_error2 = recon_error_pca(X_train)
    pd.DataFrame({'RP': recon_error1, 'PCA': recon_error2}, index=N_list).plot.bar(
        rot=0,
        figsize=(4,3),
        title=f'Reconstruction Error -- {data_name}',
        xlabel='Component Number'
    )
    plt.show()
    
def recon_error_rp_random(X_train, N):
    recon_error = []
    for seed in range(50):
        rp = SparseRandomProjection(n_components=N, random_state=seed)
        data_rp = rp.fit_transform(X_train)
        inverse_rp = rp.inverse_transform(data_rp)
        recon_error.append(mean_squared_error(X_train, inverse_rp))
    return list(range(50)), recon_error

def plot_rp_random(X_train, N, data_name):
    N_list, recon_error = recon_error_rp_random(X_train, N)
    plt.figure(figsize=(4,2.5))
    plt.plot(N_list, recon_error, '-')
    plt.xlabel('Random Seed')
    plt.ylabel('Reconstruction Error')
    plt.title(f'RP Reconstruction Error Randomness -- {data_name}')
    plt.show()

def plot_IG(X_train_train, y_train_train, X_val, y_val, data_name):
    # fit tree
    dt = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_leaf=10)
    dt.fit(X_train_train, y_train_train)
    # get Information Gain
    IG = dt.feature_importances_
    component_list = IG.argsort()[::-1] + 1
    IG = sorted(IG, reverse=True)
    # iter N
    f1 = []
    for N in range(1, len(component_list)+1):
        dt = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_leaf=10)
        dt.fit(X_train_train.iloc[:, (component_list[:N]-1)], y_train_train)
        preds = dt.predict(X_train_train.iloc[:, (component_list[:N]-1)])
        f1.append(f1_score(y_train_train, preds, average='weighted'))
    # plot
    fig, ax1 = plt.subplots(figsize=(4,2.5))
    l1 = ax1.bar(range(len(IG)), IG, label='IG')
    ax1.set_ylabel('Information Gain')
    ax1.set_xlabel('Feature Index')
    ax2 = ax1.twinx()
    l2, = ax2.plot(range(len(IG)), f1, 'k--o', label='F1')
    ax2.set_ylabel('Weighted F1 Score')
    plt.xticks(range(len(IG)), component_list)
    plt.title(f'Information Gain Plot -- {data_name}')
    plt.legend(handles=[list(l1)[0], l2], labels=['IG', 'F1'], loc='center right')
    plt.show()
    
def plot_TSNE(X_train, labels, cmap, legend_labels, data_name, algorithm_name, transform=False, legend_title='label'):
    if transform:
        tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
        projections = tsne.fit_transform(X_train)
    else:
        projections = X_train.iloc[:, [0,1]].values
    plt.figure(figsize=(5,3))
    scatter = plt.scatter(projections[:,0], projections[:,1], c=labels, s=10, cmap=cmap)
    plt.xlabel('Dimention 1')
    plt.ylabel('Dimention 2')
    plt.legend(handles=scatter.legend_elements()[0], 
               markerscale=0.5,
               labels=legend_labels,
               title=legend_title)
    plt.title(f'TSNE plot {algorithm_name} -- {data_name}')
    
def dummy_clf(X_train_train, y_train_train, X_val, y_val):
    dt = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_leaf=10)
    t_start = time.time()
    for i in range(100):
        dt.fit(X_train_train, y_train_train)
    t = (time.time() - t_start)/100
    f1_train = f1_score(y_train_train, dt.predict(X_train_train), average='weighted')
    f1_val = f1_score(y_val, dt.predict(X_val), average='weighted')
    return t, f1_train, f1_val

def plot_dummy_clf_result(t_train_list, f1_train_list, f1_val_list, names, data_name, test_label='validation'):
    N = len(t_train_list)
    
    # training time
    plt.figure(figsize=(4,3))
    plt.bar(range(N), t_train_list)
    plt.xticks(range(N), names)
    # plt.gca().set_xticklabels(names, rotation=30)
    plt.ylabel('Training Time (Sec)')
    plt.title(f'Training Time -- {data_name}')
    plt.show()
    
    # F1
    fig, ax = plt.subplots(figsize=(4,3))
    
    f1_df = pd.DataFrame({'train': f1_train_list, test_label: f1_val_list}, index=names)
    f1_df.plot.bar(ax=ax, ylabel='F1 Score (Weighted)', title=f'F1 Score -- {data_name}', zorder=3)
    plt.grid(zorder=0)
    plt.xticks(range(N), names)
    ax.set_xticklabels(names, rotation=0)
    plt.ylim(0,1)
    plt.legend(loc='upper center', ncols=2)
    plt.show()
    
def model_f1(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)
    f1_train = f1_score(y_train, pred_train, average="weighted")
    f1_test = f1_score(y_test, pred_test, average="weighted")
    return f1_train, f1_test

def compare_iter(iter_list, labels, data_name):
    N = len(iter_list)   
    plt.figure(figsize=(4,3))
    plt.bar(range(N), iter_list)
    plt.xticks(range(N), labels)
    plt.ylabel('Iteration')
    plt.title(f'Iteration to Learn -- {data_name}')
    plt.show()
