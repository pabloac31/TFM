import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, make_scorer, recall_score, precision_score
from sklearn.metrics import auc, plot_roc_curve

from collections import defaultdict

SEED = 20896  # for reproducibility


# prints a summary of the data
def print_summary(X, y):
  table = pd.DataFrame()

  VTE = X.loc[np.where(y==1)[0]]
  noVTE = X.loc[np.where(y==0)[0]]
  len_VTE, len_noVTE = len(VTE), len(noVTE)

  VTE_n, noVTE_n = [len_VTE], [len_noVTE]
  VTE_perc, noVTE_perc = [1], [1]
  col_names = ['N']

  for col in X.columns:
    if col == 'edatDx':  # compute mean
      VTE_n.append(VTE.loc[:,col].mean())
      VTE_perc.append(VTE.loc[:,col].std())
      noVTE_n.append(noVTE.loc[:,col].mean())
      noVTE_perc.append(noVTE.loc[:,col].std())
      col_names.append(col)

    # elif col == 'estadiGrup':
    #   n1 = len(VTE.loc[VTE[col] == 1])
    #   VTE_n.append(n1)
    #   VTE_perc.append(n1 / len_VTE)
    #   n2 = len(noVTE.loc[noVTE[col] == 1])
    #   noVTE_n.append(n2)
    #   noVTE_perc.append(n2 / len_noVTE)
    #   col_names.append(col + ' I')
    #   n1 = len(VTE.loc[VTE[col] == 2])
    #   VTE_n.append(n1)
    #   VTE_perc.append(n1 / len_VTE)
    #   n2 = len(noVTE.loc[noVTE[col] == 2])
    #   noVTE_n.append(n2)
    #   noVTE_perc.append(n2 / len_noVTE)
    #   col_names.append(col + ' II')
    #   n1 = len(VTE.loc[VTE[col] == 3])
    #   VTE_n.append(n1)
    #   VTE_perc.append(n1 / len_VTE)
    #   n2 = len(noVTE.loc[noVTE[col] == 3])
    #   noVTE_n.append(n2)
    #   noVTE_perc.append(n2 / len_noVTE)
    #   col_names.append(col + ' III')
    #   n1 = len(VTE.loc[VTE[col] == 4])
    #   VTE_n.append(n1)
    #   VTE_perc.append(n1 / len_VTE)
    #   n2 = len(noVTE.loc[noVTE[col] == 4])
    #   noVTE_n.append(n2)
    #   noVTE_perc.append(n2 / len_noVTE)
    #   col_names.append(col + ' IV')

    elif col == 'fumador':
      n1 = len(VTE.loc[VTE[col] == 0])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 0])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' - nunca')
      n1 = len(VTE.loc[VTE[col] == 1])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 1])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' - exfumador')
      n1 = len(VTE.loc[VTE[col] == 2])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 2])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' - fumador')

    elif col == 'hemoglobina':
      n1 = len(VTE.loc[VTE[col] < 10])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] < 10])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' <100g/L')

    elif col == 'leucocits':
      n1 = len(VTE.loc[VTE[col] > 11e3])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] > 11e3])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' >11e9/L')

    elif col == 'plaquetes':
      n1 = len(VTE.loc[VTE[col] > 350e3])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] > 350e3])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' >350e6/L')

    elif col.startswith('rs'):
      n1 = len(VTE.loc[VTE[col] == 0])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 0])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' - 0 risk alleles')
      n1 = len(VTE.loc[VTE[col] == 1])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 1])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' - 1 risk allele')

      if col not in ['rs6025','rs121909548','rs2232698']:
        n1 = len(VTE.loc[VTE[col] == 2])
        VTE_n.append(n1)
        VTE_perc.append(n1 / len_VTE)
        n2 = len(noVTE.loc[noVTE[col] == 2])
        noVTE_n.append(n2)
        noVTE_perc.append(n2 / len_noVTE)
        col_names.append(col + ' - 2 risk alleles')

    elif col == 'khorana':
        pass

    else:
      n1 = len(VTE.loc[VTE[col] == 1])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 1])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col)

  table['Variable'] = col_names
  table['VTE (n)'] = list(map(int, VTE_n))
  table['VTE (%)'] = [round(x*100,1) for x in VTE_perc]
  table['No-VTE (n)'] = list(map(int, noVTE_n))
  table['No-VTE (%)'] = [round(x*100,1) for x in noVTE_perc]

  print()
  print(table)


# print a correlation heatmap of the variables in the dataset
def corr_heatmap(df, figsize=(35,35)):
    correlations = df.corr()
    ## Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=figsize)
    fig = sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.1f',square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    fig.set_xticklabels(fig.get_xticklabels(), rotation = 90, fontsize = 10)
    fig.set_yticklabels(fig.get_yticklabels(), rotation = 0, fontsize = 10)
    plt.tight_layout()
    plt.show()



# generate a dataframe with the scores
def generate_scores_df(scores):
    names = []
    means = []
    CIs = []

    for score in scores:
        names.append(score)
        # The underlying assumption in this code is that the scores are distributed according to the Normal Distribution.
        # Then the 95% confidence interval is given by mean+/- 2*std
        mean, std = np.mean(scores[score]), np.std(scores[score])
        means.append(round(mean,2))
        CIs.append("("+str(max(round(mean-2*std,2),0))+","+str(min(round(mean+2*std,2),1))+")")

    table = pd.DataFrame()
    table['score'] = names
    table['mean'] = means
    table['95% CI'] = CIs

    return table


# obtain the metrics for the Khorana score
def test_khorana(khorana, y):
    # High risk: Khorana >= 3
    pred_khorana = khorana >= 3

    auc_khorana = roc_auc_score(y, pred_khorana)
    print("AUC: ", round(auc_khorana*100, 2))

    acc_khorana = sum(pred_khorana == y) / len(y)
    print("Accuracy (%):", round(acc_khorana*100, 2))

    tn, fp, fn, tp = confusion_matrix(y, pred_khorana).ravel()
    confusion_matrix(y, pred_khorana)
    sensivity_khorana = tp / (tp+fn)
    specificity_khorana = tn / (fp+tn)
    PPV_khorana = tp / (tp+fp)
    NPV_khorana = tn / (fn+tn)
    print("Sensivity (%):", round(sensivity_khorana*100,2))
    print("Specificity (%):", round(specificity_khorana,4)*100)
    print("PPV (%):", round(PPV_khorana,4)*100)
    print("NPV (%):", round(NPV_khorana,4)*100)


# Test the Khorana score using the bootstrap approach (as in the paper)
def test_khorana_bootstrap(khorana, y, n=100):
    scores = defaultdict(list)
    np.random.seed(SEED)

    for i in range(n):
        idx = np.random.choice(len(y), len(y))
        khorana_bt = khorana[idx]
        y_bt = y[idx]

        # High risk: Khorana >= 3
        pred_khorana = khorana_bt >= 3

        auc_khorana = roc_auc_score(y_bt, pred_khorana)
        scores['AUC'].append(auc_khorana)

        acc_khorana = sum(pred_khorana == y_bt) / len(y_bt)
        scores['accuracy'].append(acc_khorana)

        tn, fp, fn, tp = confusion_matrix(y_bt, pred_khorana).ravel()
        sensivity_khorana = tp / (tp+fn)
        specificity_khorana = tn / (fp+tn)
        PPV_khorana = tp / (tp+fp)
        NPV_khorana = tn / (fn+tn)
        scores['sensitivity'].append(sensivity_khorana)
        scores['specificity'].append(specificity_khorana)
        scores['PPV'].append(PPV_khorana)
        scores['NPV'].append(NPV_khorana)

    return generate_scores_df(scores)


# test a model using 10-fold cross validation
def test_model(clf, X, y, cutoff=0.8):
    if not isinstance(X,np.ndarray):
        X = X.to_numpy()

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    tprs = []
    aucs = []
    scores = defaultdict(list)
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X[train], y[train])
        viz = plot_roc_curve(clf, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        if hasattr(clf, 'predict_proba'):
            y_pred_test = clf.predict_proba(X[test])[:,1]
        else:
            y_pred_test = clf.decision_function(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], y_pred_test)
        scores['AUC'].append(auc(fpr,tpr))

        # thresholds (specificity ~ 80%, as indicated in the paper)
        idx = np.argmin(abs((1-fpr) - cutoff))
        threshold = thresholds[idx]

        # high risk: TiC-Onco risk >= threshold
        y_hat_test = y_pred_test >= threshold
        acc = sum(y_hat_test == y[test]) / len(y[test])
        scores['accuracy'].append(acc)

        tn, fp, fn, tp = confusion_matrix(y[test], y_hat_test).ravel()
        sensitivity = tp / (tp+fn)
        specificity = tn / (fp+tn)
        PPV = tp / (tp+fp) if tp+fp > 0 else 0
        NPV = tn / (fn+tn)
        scores['sensitivity'].append(sensitivity)
        scores['specificity'].append(specificity)
        scores['PPV'].append(PPV)
        scores['NPV'].append(NPV)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic curves")
    ax.legend(loc="lower right", bbox_to_anchor=(1.63, 0))
    plt.show()

    return generate_scores_df(scores)



# test a model using the bootstrap approach
def test_model_bootstrap(clf, X, y, n=100, cutoff=0.8):

    X = X.to_numpy()
    np.random.seed(SEED)

    tprs = []
    aucs = []
    scores = defaultdict(list)

    for i in range(n):
        idx = np.random.choice(len(X), len(X))
        X_bt = X[idx,:]
        y_bt = y[idx]
        clf.fit(X_bt, y_bt)

        if hasattr(clf, 'predict_proba'):
            y_pred_test = clf.predict_proba(X_bt)[:,1]
        else:
            y_pred_test = clf.decision_function(X_bt)

        fpr, tpr, thresholds = roc_curve(y_bt, y_pred_test)
        scores['AUC'].append(auc(fpr,tpr))

        # thresholds (specificity ~ 80%, as indicated in the paper)
        idx = np.argmin(abs((1-fpr) - cutoff))
        threshold = thresholds[idx]

        # high risk: TiC-Onco risk >= threshold
        y_hat_test = y_pred_test >= threshold
        acc = sum(y_hat_test == y_bt) / len(y_bt)
        scores['accuracy'].append(acc)

        tn, fp, fn, tp = confusion_matrix(y_bt, y_hat_test).ravel()
        sensitivity = tp / (tp+fn)
        specificity = tn / (fp+tn)
        PPV = tp / (tp+fp) if tp+fp > 0 else 0
        NPV = tn / (fn+tn)
        scores['sensitivity'].append(sensitivity)
        scores['specificity'].append(specificity)
        scores['PPV'].append(PPV)
        scores['NPV'].append(NPV)

    return generate_scores_df(scores)
