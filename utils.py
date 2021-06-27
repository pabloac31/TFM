import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, make_scorer, recall_score, precision_score
from sklearn.metrics import auc, plot_roc_curve

from collections import defaultdict


# considered features
gen_features = ['rs6025','rs4524','rs1799963','rs1801020','rs5985','rs121909548',
                        'rs2232698','rs8176719','rs7853989','rs8176749','rs8176750']
clinical_features = ['sexe','edatDx','diabetesM','fumador','Family','bmi','dislip', 'hta_desc','khorana',
                        'tipusTumor_desc','estadiGrup','hemoglobina','plaquetes','leucocits']
target = ['caseAtVisit']


def preprocess_gen_data(gen_data):
    # Computing the number of risk alleles for each gene
    gen_data.loc[:,'rs6025'].replace(['GG','AG'], [0,1], inplace=True)
    gen_data.loc[:,'rs4524'].replace(['CC','CT','TT'], [0,1,2], inplace=True)
    gen_data.loc[:,'rs1799963'].replace(['GG','AG','AA'], [0,1,2], inplace=True)
    gen_data.loc[:,'rs1801020'].replace(['CC','CT','TT'], [0,1,2], inplace=True)
    gen_data.loc[:,'rs5985'].replace(['GG','GT','TT'], [0,1,2], inplace=True)
    gen_data.loc[:,'rs121909548'].replace(['GG','GT'], [0,1], inplace=True)
    gen_data.loc[:,'rs2232698'].replace(['CC','CT'], [0,1], inplace=True)

    # A1 blood group
    gen_data.loc[:,'rs8176719'].replace(['--','-G','GG'], [0,1,2], inplace=True)
    gen_data.loc[:,'rs7853989'].replace(['CC','CG','GG'], [0,1,2], inplace=True)
    gen_data.loc[:,'rs8176749'].replace(['GG','AG'], [0,1], inplace=True)
    gen_data.loc[:,'rs8176750'].replace(['CC','-C'], [0,1], inplace=True)

    gen_data.loc[:,:].reset_index(drop=True, inplace=True)

    return gen_data


def preprocess_clinical_data(clinical_data):
    clinical_data.loc[:,'sexe'].replace(['Hombre','Mujer'], [0,1], inplace=True)
    clinical_data.loc[:,'diabetesM'].replace(['No','Sí'], [0,1], inplace=True)
    clinical_data.loc[:,'fumador'].replace(['Nunca','Exfumador'], 0, inplace=True)
    clinical_data.loc[:,'fumador'].replace('Fumador activo', 1, inplace=True)
    clinical_data.loc[:,'bmi'].replace('Underweight: BMI < 18.5 Kg/m2', 0, inplace=True)
    clinical_data.loc[:,'bmi'].replace('Normal: BMI ~ 18.5-24.9 Kg/m2', 0, inplace=True)
    clinical_data.loc[:,'bmi'].replace('Overweight: BMI ~25-29.9 Kg/m2', 1, inplace=True)
    clinical_data.loc[:,'bmi'].replace('Obese: BMI > 30 kg/m2', 1, inplace=True)
    clinical_data.loc[:,'dislip'].replace(['No','Sí'], [0,1], inplace=True)
    clinical_data.loc[:,'hta_desc'].replace(['No','Sí'], [0,1], inplace=True)
    clinical_data.loc[:,'khorana'] = [1 if n>=3 else 0 for n in clinical_data['khorana']]
    clinical_data.loc[:,'tipusTumor_colon'] = [1 if t=='Cáncer colorrectal' else 0 for t in clinical_data['tipusTumor_desc']]
    clinical_data.loc[:,'tipusTumor_pancreas'] = [1 if t=='Cáncer de páncreas' else 0 for t in clinical_data['tipusTumor_desc']]
    clinical_data.loc[:,'tipusTumor_pulmon'] = [1 if t=='Cáncer de pulmón no microcítico' else 0 for t in clinical_data['tipusTumor_desc']]
    clinical_data.loc[:,'tipusTumor_esofago'] = [1 if t=='Cáncer esófago' else 0 for t in clinical_data['tipusTumor_desc']]
    clinical_data.loc[:,'tipusTumor_estomago'] = [1 if t=='Cáncer gástrico o de estómago' else 0 for t in clinical_data['tipusTumor_desc']]
    clinical_data.drop('tipusTumor_desc', axis=1, inplace=True)
    clinical_data.loc[:,'estadiGrup_I_II'] = [1 if g in ['IA','IB','IIA','IIB','IIC'] else 0 for g in clinical_data['estadiGrup']]
    clinical_data.loc[:,'estadiGrup_III'] = [1 if g in ['III','IIIA','IIIB','IIIC'] else 0 for g in clinical_data['estadiGrup']]
    clinical_data.loc[:,'estadiGrup_IV'] = [1 if g in ['IV','IVA','IVB'] else 0 for g in clinical_data['estadiGrup']]
    clinical_data.drop('estadiGrup', axis=1, inplace=True)
    clinical_data.loc[:,'hemoglobina'] = [1 if n<10 else 0 for n in clinical_data['hemoglobina']]
    clinical_data.loc[:,'plaquetes'] = [1 if n>350000 else 0 for n in clinical_data['plaquetes']]
    clinical_data.loc[:,'leucocits'] = [1 if n>11000 else 0 for n in clinical_data['leucocits']]

    clinical_data.loc[:,:].reset_index(drop=True, inplace=True)

    return clinical_data


# preprocess data from excel file (paper)
def preprocess_data(df, drop_na=True, summary=False):

    df = df[df['excluido']==0]

    df = df[gen_features + clinical_features + target]
    print("Initial shape:", df.shape)

    if drop_na:
        df = df.replace(['NoCall', 'Desconocido'], 0)
        # df['tipusTumor_desc'] = df['tipusTumor_desc'].replace('-', np.NaN)
        df[['diabetesM','dislip','hta_desc']] = df[['diabetesM','dislip','hta_desc']].replace('-',0)

        df['khorana'].fillna(0, inplace=True) ### delete

        r, _ = np.where(df.loc[:, df.columns != 'caseAtVisit'].isna())
        idx = np.unique(r)

        df.drop(idx, inplace=True)

    gen_data = preprocess_gen_data(df[gen_features])
    print("Genetic data shape:", gen_data.shape)

    clinical_data = preprocess_clinical_data(df[clinical_features])
    print("Clinical data shape:", clinical_data.shape)

    X = pd.concat([gen_data[['rs2232698','rs6025','rs5985','rs4524']], clinical_data[['bmi','Family']]], axis=1)
    X['primary_tumour_site'] = [1 if t=='Cáncer de pulmón no microcítico' else 2 if t in ['Cáncer gástrico o de estómago','Cáncer de páncreas'] else 0 for t in df['tipusTumor_desc']]
    X['tumour_stage'] = [1 if t in ['IV','IVA','IVB'] else 0 for t in df['estadiGrup']]
    X.reset_index(drop=True, inplace=True)
    print("Clinical-Genetic data shape:", X.shape)

    y = np.array([1 if t in [0,1] else 0 for t in df['caseAtVisit']])
    print("Target shape:", y.shape)

    unique_elements, counts_elements = np.unique(y, return_counts=True)
    print("\nNumber of No-VTE (0) and VTE (1):", counts_elements)

    if summary:  # TODO: add gen_data
        table = pd.DataFrame()
        VTE_n, noVTE_n = [], []
        VTE_perc, noVTE_perc = [], []

        VTE = clinical_data.loc[np.where(y==1)[0]]
        noVTE = clinical_data.loc[np.where(y==0)[0]]
        len_VTE, len_noVTE = len(VTE), len(noVTE)

        for col in clinical_data.columns:
            if col == 'edatDx':  # compute mean
                VTE_n.append(VTE.loc[:,col].mean())
                VTE_perc.append(VTE.loc[:,col].std())
                noVTE_n.append(noVTE.loc[:,col].mean())
                noVTE_perc.append(noVTE.loc[:,col].std())

            else:
                n1 = len(VTE.loc[VTE[col] == 1])
                VTE_n.append(n1)
                VTE_perc.append(n1 / len_VTE)
                n2 = len(noVTE.loc[noVTE[col] == 1])
                noVTE_n.append(n2)
                noVTE_perc.append(n2 / len_noVTE)

        table['Variable'] = clinical_data.columns
        table['VTE (n)'] = list(map(int, VTE_n))
        table['VTE (%)'] = [round(x*100,1) for x in VTE_perc]
        table['No-VTE (n)'] = list(map(int, noVTE_n))
        table['No-VTE (%)'] = [round(x*100,1) for x in noVTE_perc]

        print()
        print(table)

    return gen_data, clinical_data, X, y


# Compute p-values usign permutation test
def perm_test(data, labels, model, n=500):

  # Logistic regression coefficients
  coef = model.fit(data, labels).coef_[0]

  coefs = np.zeros((n, len(coef)))
  p_values = []

  for i in range(n):
    X_perm = shuffle(data) # permutation
    model.fit(X_perm, labels)
    coefs[i] = model.coef_[0]

  for i in range(len(coef)):
    p_value = len(np.where(np.abs(coefs[:,i]) >= np.abs(coef[i]))[0]) / n
    p_values.append(p_value)

  return p_values


#####################################################
#####################################################

def get_data(df, exclude=False):
  df = df[gen_features + clinical_features + target + ['excluido']]

  print("Initial shape:", df.shape)

  # obtain NaNs
  df = df.replace(['NoCall', 'Desconocido'], np.NaN)
  df['tipusTumor_desc'] = df['tipusTumor_desc'].replace('-', np.NaN)
  df['diabetesM'] = df['diabetesM'].replace('-','No')
  df['dislip'] = df['dislip'].replace('-','No')
  df['hta_desc'] = df['hta_desc'].replace('-','No')

  # drop NaN values
  r, _ = np.where(df.loc[:, df.columns != 'caseAtVisit'].isna())
  idx = np.unique(r)
  df.drop(idx, inplace=True)

  if exclude:
    df = df[df['excluido']==0]

  df.drop('excluido', axis=1, inplace=True)

  # Computing the number of risk alleles for each gene
  df.loc[:,'rs6025'].replace(['GG','AG'], [0,1], inplace=True)
  df.loc[:,'rs4524'].replace(['CC','CT','TT'], [0,1,2], inplace=True)
  df.loc[:,'rs1799963'].replace(['GG','AG','AA'], [0,1,2], inplace=True)
  df.loc[:,'rs1801020'].replace(['CC','CT','TT'], [0,1,2], inplace=True)
  df.loc[:,'rs5985'].replace(['GG','GT','TT'], [0,1,2], inplace=True)
  df.loc[:,'rs121909548'].replace(['GG','GT'], [0,1], inplace=True)
  df.loc[:,'rs2232698'].replace(['CC','CT'], [0,1], inplace=True)
  # A1 blood group
  df.loc[:,'rs8176719'].replace(['--','-G','GG'], [0,1,2], inplace=True)
  df.loc[:,'rs7853989'].replace(['CC','CG','GG'], [0,1,2], inplace=True)
  df.loc[:,'rs8176749'].replace(['GG','AG'], [0,1], inplace=True)
  df.loc[:,'rs8176750'].replace(['CC','-C'], [0,1], inplace=True)

  # Preprocess clinical variables
  df.loc[:,'sexe'].replace(['Hombre','Mujer'], [0,1], inplace=True)
  df.loc[:,'diabetesM'].replace(['No','Sí'], [0,1], inplace=True)
  df.loc[:,'fumador'].replace(['Nunca','Exfumador'],0, inplace=True)
  df.loc[:,'fumador'].replace('Fumador activo',1, inplace=True)
  df.loc[:,'bmi'].replace(['Underweight: BMI < 18.5 Kg/m2','Normal: BMI ~ 18.5-24.9 Kg/m2'], 0, inplace=True)
  df.loc[:,'bmi'].replace(['Overweight: BMI ~25-29.9 Kg/m2','Obese: BMI > 30 kg/m2'], 1, inplace=True)
  df.loc[:,'dislip'].replace(['No','Sí'], [0,1], inplace=True)
  df.loc[:,'hta_desc'].replace(['No','Sí'], [0,1], inplace=True)
  # df.loc[:,'khorana'] = [1 if n>=3 else 0 for n in clinical_data['khorana']]
  df['tipusTumor_colon'] = [1 if t=='Cáncer colorrectal' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_pancreas'] = [1 if t=='Cáncer de páncreas' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_pulmon'] = [1 if t=='Cáncer de pulmón no microcítico' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_esofago'] = [1 if t=='Cáncer esófago' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_estomago'] = [1 if t=='Cáncer gástrico o de estómago' else 0 for t in df['tipusTumor_desc']]
  df.drop('tipusTumor_desc', axis=1, inplace=True)
  df['estadiGrup_I_II'] = [1 if g in ['IA','IB','IIA','IIB','IIC'] else 0 for g in df['estadiGrup']]
  df['estadiGrup_III'] = [1 if g in ['III','IIIA','IIIB','IIIC'] else 0 for g in df['estadiGrup']]
  df['estadiGrup_IV'] = [1 if g in ['IV','IVA','IVB'] else 0 for g in df['estadiGrup']]
  df.drop('estadiGrup', axis=1, inplace=True)
#   df.loc[:,'estadiGrup'].replace(['IA','IB'],1, inplace=True)
#   df.loc[:,'estadiGrup'].replace( ['IIA','IIB','IIC'],2, inplace=True)
#   df.loc[:,'estadiGrup'].replace(['III','IIIA','IIIB','IIIC'],3, inplace=True)
#   df.loc[:,'estadiGrup'].replace(['IV','IVA','IVB'],4, inplace=True)

  df.drop('rs7853989', axis=1, inplace=True) # high correlated with rs8176749

  X = df[df.columns.difference(target + ['khorana'])]
  X.reset_index(drop=True, inplace=True)
  print("Features shape:", X.shape)

  y = np.array([1 if t in [0,1] else 0 for t in df['caseAtVisit']])
  print("Target shape:", y.shape)

  khorana = df['khorana']

  unique_elements, counts_elements = np.unique(y, return_counts=True)
  print("\nNumber of No-VTE (0) and VTE (1):", counts_elements)

  return X, y, khorana


def print_summary(X, y):
  table = pd.DataFrame()
  VTE_n, noVTE_n = [], []
  VTE_perc, noVTE_perc = [], []
  col_names = []

  VTE = X.loc[np.where(y==1)[0]]
  noVTE = X.loc[np.where(y==0)[0]]
  len_VTE, len_noVTE = len(VTE), len(noVTE)

  for col in X.columns:
    if col == 'edatDx':  # compute mean
      VTE_n.append(VTE.loc[:,col].mean())
      VTE_perc.append(VTE.loc[:,col].std())
      noVTE_n.append(noVTE.loc[:,col].mean())
      noVTE_perc.append(noVTE.loc[:,col].std())
      col_names.append(col)

    elif col == 'estadiGrup':
      n1 = len(VTE.loc[VTE[col] == 1])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 1])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' I')
      n1 = len(VTE.loc[VTE[col] == 2])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 2])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' II')
      n1 = len(VTE.loc[VTE[col] == 3])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 3])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' III')
      n1 = len(VTE.loc[VTE[col] == 4])
      VTE_n.append(n1)
      VTE_perc.append(n1 / len_VTE)
      n2 = len(noVTE.loc[noVTE[col] == 4])
      noVTE_n.append(n2)
      noVTE_perc.append(n2 / len_noVTE)
      col_names.append(col + ' IV')

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


def run_exps(models, X_train, y_train):
    dfs = []
    results = []
    names = []
    scoring = ['roc_auc', 'precision', 'balanced_accuracy']
    target_names = ['noVTE', 'VTE']

    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2037)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, return_train_score=False)
        # clf = model.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print(name)
        # print(classification_report(y_test, y_pred, target_names=target_names))

        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)

    final = pd.concat(dfs, ignore_index=True)
    return final

def results_bootstrap(final):
  bootstraps = []
  for model in list(set(final.model.values)):
    model_df = final.loc[final.model == model]
    bootstrap = model_df.sample(n=50, replace=True)
    bootstraps.append(bootstrap)

  bootstrap_df = pd.concat(bootstraps, ignore_index=True)
  results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')

  time_metrics = ['fit_time','score_time']

  ## PERFORMANCE METRICS
  results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
  results_long_nofit = results_long_nofit.sort_values(by='values')

  return bootstrap_df, results_long_nofit

def plot_performances(results_long_nofit):
  # plot performance metrics from the 5-fold cross validation
  plt.figure(figsize=(15, 8))
  sns.set(font_scale=1.5)
  g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.title('Comparison of Model by Classification Metric')
  # plt.savefig('./benchmark_models_performance.png',dpi=300)
  plt.show()

def tabulate_results(bootstrap_df, results_long_nofit):
  metrics = list(set(results_long_nofit.metrics.values))
  return bootstrap_df.groupby(['model'])[metrics].agg([np.mean, np.std])


def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):

  # compute roc curve
  fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
  fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_test)
  roc_auc = auc(fpr, tpr)
  roc_auc2 = auc(fpr2, tpr2)
  print("AUC score (train):", round(roc_auc,4))
  print("AUC score (test):", round(roc_auc2,4))

  plt.title('Receiver Operating Characteristic (test)')
  plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc2)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

  # thresholds (specificity ~ 80%, as indicated in the paper)
  idx = np.argmin(abs((1-fpr) - 0.8))
  idx2 = np.argmin(abs((1-fpr2) - 0.8))
  threshold = thresholds[idx]
  threshold2 = thresholds2[idx2]

  # high risk: TiC-Onco risk >= threshold
  y_hat_train = y_pred_train >= threshold
  y_hat_test = y_pred_test >= threshold2
  acc = sum(y_hat_train == y_train) / len(y_train)
  acc2 = sum(y_hat_test == y_test) / len(y_test)
  print("\nAccuracy in train set (%):", round(acc*100, 2))
  print("Accuracy in test set (%):", round(acc2*100, 2))

  # confusion matrix
  for subset, y, y_hat in zip(['Train set','Test set'], [y_train,y_test], [y_hat_train,y_hat_test]):
    print("\n=====" + subset + "=====")
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    print(confusion_matrix(y, y_hat))
    print()

    sensivity = tp / (tp+fn)
    specificity = tn / (fp+tn)
    PPV = tp / (tp+fp)
    NPV = tn / (fn+tn)

    print("Sensivity (%):", round(sensivity*100,2))
    print("Specificity (%):", round(specificity,4)*100)
    print("Precision (%):", round(PPV,4)*100)
    print("NPV (%):", round(NPV,4)*100)


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


def test_model(clf, X, y, cutoff=0.8):
    if not isinstance(X,np.ndarray):
        X = X.to_numpy()
    # scoring = {'AUC': make_scorer(roc_auc_score),
    #            'sensitivity': make_scorer(recall_score),
    #            'specificity': make_scorer(recall_score, pos_label=0),
    #            'PPV': make_scorer(precision_score),
    #            'NPV': make_scorer(precision_score, pos_label=0)
    #             }

    # 10-fold CV
    # scores = cross_validate(clf, X, y, scoring=scoring, cv=10)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=10)

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


def test_khorana_bootstrap(khorana, y, n=100):
    scores = defaultdict(list)

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


def test_model_bootstrap(clf, X, y, n=100, cutoff=0.8):
    X = X.to_numpy()

    cv = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    scores = defaultdict(list)
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(n):
        idx = np.random.choice(len(X), len(X))
        X_bt = X[idx,:]
        y_bt = y[idx]
        clf.fit(X_bt, y_bt)

        y_pred_test = clf.predict_proba(X_bt)[:,1]
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
