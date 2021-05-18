import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, confusion_matrix

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

def get_data(df):
  df = df[gen_features + clinical_features + target]
  print("Initial shape:", df.shape)

  # remove NaNs
  df = df.replace(['NoCall', 'Desconocido'], np.NaN)
  df['tipusTumor_desc'] = df['tipusTumor_desc'].replace('-', np.NaN)
  df[['diabetesM','dislip','hta_desc']] = df[['diabetesM','dislip','hta_desc']].replace('-','No')

  r, _ = np.where(df.loc[:, df.columns != 'caseAtVisit'].isna())
  idx = np.unique(r)

  df.drop(idx, inplace=True)

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
  df.loc[:,'fumador'].replace(['Nunca','Exfumador'], [0,1], inplace=True) ##
  df.loc[:,'fumador'].replace('Fumador activo', 2, inplace=True) ##
  df.loc[:,'bmi'].replace('Underweight: BMI < 18.5 Kg/m2', 0, inplace=True)
  df.loc[:,'bmi'].replace('Normal: BMI ~ 18.5-24.9 Kg/m2', 1, inplace=True) ##
  df.loc[:,'bmi'].replace('Overweight: BMI ~25-29.9 Kg/m2', 2, inplace=True) ##
  df.loc[:,'bmi'].replace('Obese: BMI > 30 kg/m2', 3, inplace=True) ##
  df.loc[:,'dislip'].replace(['No','Sí'], [0,1], inplace=True)
  df.loc[:,'hta_desc'].replace(['No','Sí'], [0,1], inplace=True)
  # df.loc[:,'khorana'] = [1 if n>=3 else 0 for n in clinical_data['khorana']]
  df['tipusTumor_colon'] = [1 if t=='Cáncer colorrectal' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_pancreas'] = [1 if t=='Cáncer de páncreas' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_pulmon'] = [1 if t=='Cáncer de pulmón no microcítico' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_esofago'] = [1 if t=='Cáncer esófago' else 0 for t in df['tipusTumor_desc']]
  df['tipusTumor_estomago'] = [1 if t=='Cáncer gástrico o de estómago' else 0 for t in df['tipusTumor_desc']]
  df.drop('tipusTumor_desc', axis=1, inplace=True)
  df['estadiGrup_I'] = [1 if g in ['IA','IB'] else 0 for g in df['estadiGrup']] ##
  df['estadiGrup_II'] = [1 if g in ['IIA','IIB','IIC'] else 0 for g in df['estadiGrup']] ##
  df['estadiGrup_III'] = [1 if g in ['III','IIIA','IIIB','IIIC'] else 0 for g in df['estadiGrup']]
  df['estadiGrup_IV'] = [1 if g in ['IV','IVA','IVB'] else 0 for g in df['estadiGrup']]
  df.drop('estadiGrup', axis=1, inplace=True)

  X = df[df.columns.difference(target)]
  X.reset_index(drop=True, inplace=True)
  print("Features shape:", X.shape)

  y = np.array([1 if t in [0,1] else 0 for t in df['caseAtVisit']])
  print("Target shape:", y.shape)

  unique_elements, counts_elements = np.unique(y, return_counts=True)
  print("\nNumber of No-VTE (0) and VTE (1):", counts_elements)

  return X, y


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
