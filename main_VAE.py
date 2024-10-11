import pandas as pd
import numpy as np
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import compress

os.chdir("/home/tpjoe/tpjoe@stanford.edu/project_TPN/Python/paper_github")
path = "/home/tpjoe/tpjoe@stanford.edu/project_TPN/"

from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import copy

from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model import *
from utils import *

#################################################
# define variable list
input_var = ["gest_age", "bw",
            'gender_concept_id_8507', 'gender_concept_id_8532',  
            'race_concept_id_8515', 'race_concept_id_8516',  
            'race_concept_id_8527', 'race_concept_id_8557', 'race_concept_id_8657',
            
            'Alb_lab_value', 'Ca_lab_value', 'Cl_lab_value', 'CO2_lab_value', 
            'Glu_lab_value', 'K_lab_value', 'Mg_lab_value', 'Na_lab_value', 
            'PO4_lab_value', 'BUN_lab_value', 'Cr_lab_value', 'Tri_lab_value', 
            'ALKP_lab_value', 'ALT_lab_value', 'AST_lab_value', 'CaI_lab_value', 
            "TodaysWeight", "day_since_birth", "max_chole_TPNEHR",
            
            "VTBI", "FluidDose", "EnteralTotal", 
            "TPNHours",
            "ProtocolName_NEONATAL", # "ProtocolName_PEDIATRIC",
            "LineID_2", #"LineID_1",
            'FatProduct_SMOFlipid_20', 'FatProduct_Intralipid_20', 'FatProduct_Omegaven_10'
            ] + ['encoded_'+str(i) for i in range(32)]
vars_3 = ['FatDose', 'AADose', 'DexDose', 'Acetate', 'Calcium', 'Copper', 'Famotidine', 'Heparin', 'Levocar',           
          'Magnesium', 'MVIDose', 'Phosphate', 'Potassium', 'Selenium', 'Sodium', 'Zinc', 'Chloride']
#################################################

def prepare_data(data, input_var, vars_3, meta_var):
    X_ = data.set_index('sMRN', drop=False).loc[:, input_var].copy()
    y_ = data.set_index('sMRN', drop=False).loc[:, vars_3].copy()
    meta_ = data.set_index('sMRN', drop=False).loc[:, meta_var].copy()
    X_list = [torch.Tensor(X_.loc[i, :].to_numpy()) for i in X_.index.unique()]
    y_list = [torch.Tensor(y_.loc[i, :].to_numpy()) for i in y_.index.unique()]
    meta_list = [torch.Tensor(meta_.loc[i, :].to_numpy()) for i in meta_.index.unique()]
    X_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in X_list]
    y_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in y_list]
    meta_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in meta_list]
    return X_list, y_list, meta_list, X_, y_, meta_


def create_masks(data_test, ndays, cv, path):
    # mrn_randomized = pd.read_csv(path + "data/cv_fold_id_" + str(ndays) + "day.csv")
    mrn_randomized = pd.read_csv(path + 'Python/paper_github/mock_data/mock_TrainTest.csv')
    mrn_randomized['final'] = mrn_randomized[cv].apply(lambda x: 'Test' if ((x[cv[0]]=='Test') | (x[cv[1]]=='Test')) else \
                                                                 'Val' if ((x[cv[0]]=='Val')) else \
                                                                 'Train', axis=1)
    mrn_cv = data_test.reset_index().merge(mrn_randomized, left_on='MedicalRecordNum', right_on='shuffled_mrn')[['final', 'MedicalRecordNum']]
    mrn_cv.set_index('MedicalRecordNum', inplace=True, drop=False)
    test_mrn = mrn_cv.loc[mrn_cv.iloc[:, 0]=='Test', 'MedicalRecordNum'].unique()
    val_mrn = mrn_cv.loc[mrn_cv.iloc[:, 0]=='Val', 'MedicalRecordNum'].unique()
    train_mrn = mrn_cv.loc[mrn_cv.iloc[:, 0]=='Train', 'MedicalRecordNum'].unique()
    return train_mrn, val_mrn, test_mrn


def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    input_scaler = StandardScaler()
    X_train = input_scaler.fit_transform(X_train)
    X_val = input_scaler.transform(X_val)
    X_test = input_scaler.transform(X_test)
    
    output_scaler = StandardScaler()
    y_train = output_scaler.fit_transform(y_train)
    y_val = output_scaler.transform(y_val)
    y_test = output_scaler.transform(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def convert_to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, device):
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, criterion, optimizer, train_dataloader, X_val, y_val, device, epochs=1000, patience=5, delta=0.01):
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(epochs):
        epoch_trainloss = 0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X_train_, y_train_ = batch[0].to(device), batch[1].to(device)
            y_pred = model(X_train_)
            train_loss = criterion(y_pred, y_train_)
            epoch_trainloss += float(train_loss.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = criterion(y_pred_val, y_val)
        val_loss = float(val_loss.detach().cpu().numpy())
        val_r = np.nanmean([pearsonr(y_pred_val.detach().cpu().numpy()[:, i], y_val.detach().cpu().numpy()[:, i])[0] for i in range(y_pred_val.shape[1])])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if early_stopping(val_loss):
            print("Early stopping!")
            break
        print(f"Epoch {epoch}, Train Loss: {(epoch_trainloss)/len(train_dataloader)}, Val Loss: {val_loss}, Val R: {val_r}")
    
    return best_model_state


def main_training_loop(X_train, y_train, X_val, y_val, latent_dim, device='cpu'):
    input_size = X_train.shape[1]
    output_size = 1
    n_tasks = y_train.shape[1]
    model = VAE(input_size, latent_dim, output_size, n_tasks).to(device)
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Create dataset and DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    # Train the model
    best_model_state = train_model(model, criterion, optimizer, train_dataloader, X_val, y_val, device)
    # Load the best model state
    model.load_state_dict(best_model_state)
    return model


def get_latent_rep(model, X_train, X_val, X_test):
    model.eval()
    y_pred_train = model.encoding(X_train).detach().cpu().numpy()
    y_pred_val = model.encoding(X_val).detach().cpu().numpy()
    y_pred_test = model.encoding(X_test).detach().cpu().numpy()
    return y_pred_train, y_pred_val, y_pred_test


def map_columns(data_test):
    data_test['LineID'] = data_test['LineID_2'].copy().map({0: 1, 1: 2})
    data_test['ProtocolName'] = data_test['ProtocolName_NEONATAL'].copy().map({0: 'PEDIATRIC', 1: 'NEONATAL'})
    return data_test


def create_df_with_latent(data_test, y_pred_train, y_pred_val, y_pred_test, y_train, y_val, y_test, train_mrn, val_mrn, test_mrn, latent_dim, vars_3):
    test_data = pd.concat([data_test.loc[data_test.MedicalRecordNum.isin(test_mrn), :].reset_index()[['MedicalRecordNum', 'LineID', 'ProtocolName', 'OrderNum']],
                           pd.DataFrame(y_pred_test, columns=['latent_' + str(i) for i in range(latent_dim)]),
                           pd.DataFrame(y_test.detach().cpu().numpy(), columns=vars_3)], axis=1)
    train_data = pd.concat([data_test.loc[data_test.MedicalRecordNum.isin(train_mrn), :].reset_index()[['MedicalRecordNum', 'LineID', 'ProtocolName', 'OrderNum']],
                            pd.DataFrame(y_pred_train, columns=['latent_' + str(i) for i in range(latent_dim)]),
                            pd.DataFrame(y_train.detach().cpu().numpy(), columns=vars_3)], axis=1)
    val_data = pd.concat([data_test.loc[data_test.MedicalRecordNum.isin(val_mrn), :].reset_index()[['MedicalRecordNum', 'LineID', 'ProtocolName', 'OrderNum']],
                          pd.DataFrame(y_pred_val, columns=['latent_' + str(i) for i in range(latent_dim)]),
                          pd.DataFrame(y_val.detach().cpu().numpy(), columns=vars_3)], axis=1)
    all_data = pd.concat([train_data, val_data, test_data], axis=0).reset_index(drop=True)
    return train_data, val_data, test_data, all_data


class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss > self.best_val_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss didn't improve for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def perform_clustering(train_data_, test_data_, latent_dim, model, device, vars_3, central_k, nk_max, nk_min):
    def process_y(y, dist_matrix, n_group, km, train_data_, latent_dim, model, device, vars_3):
        sorted_dist_matrix = np.sort(dist_matrix.to_numpy()[np.triu_indices(n=n_group, k=1)])
        if y < len(sorted_dist_matrix):
            to_merge = (dist_matrix == sorted_dist_matrix[y])
            to_merge = to_merge[to_merge].stack().index.values[0]

            centers_new = km.cluster_centers_.copy()
            centers_new.loc[to_merge[0], :] = train_data_.loc[:,
                ['latent_' + str(i) for i in range(latent_dim)]].loc[np.isin(km.labels_, to_merge)].mean().values
            centers_new.loc[to_merge[1], :] = train_data_.loc[:,
                ['latent_' + str(i) for i in range(latent_dim)]].loc[np.isin(km.labels_, to_merge)].mean().values

            new_data_m = pd.DataFrame(pd.Series(km.labels_).apply(lambda x: centers_new.loc[x, :]))
            predictions_kbn = pd.DataFrame([i.detach().cpu().numpy().reshape(-1) for i in
                                            model.decoding(torch.Tensor(new_data_m.to_numpy()).to(device))],
                                           index=vars_3).transpose()

            cor_merged = np.mean(list(vectorized_r(torch.Tensor(train_data_[vars_3].to_numpy()), torch.Tensor(predictions_kbn[vars_3].to_numpy()))))
            # pd.Series(
                # [pearsonr(train_data_.loc[:, i], predictions_kbn.loc[:, i])[0] for i in vars_3], index=vars_3)
            return cor_dkbn.mean() - cor_merged.mean()
        else:
            return 9999  # or any value you want for cases where y > len(sorted_dist_matrix)

    ## k-means of the bottle-neck output with k=nk_max
    km = KMeans(n_clusters=nk_max, n_init=10, random_state=10)
    km.fit(train_data_.loc[:, ['latent_' + str(i) for i in range(latent_dim)]])
    km_reserve = copy.deepcopy(km)

    # train
    assn_train_ped = km.predict(train_data_.loc[:, ['latent_' + str(i) for i in range(latent_dim)]])
    centers_train_ped = pd.DataFrame(pd.Series(assn_train_ped).apply(lambda x: km.cluster_centers_[x]).tolist())
    predictions_kbn = pd.DataFrame([i.detach().cpu().numpy().reshape(-1) for i in
                                    model.decoding(torch.Tensor(centers_train_ped.to_numpy()).to(device))], index=vars_3).transpose()

    # test
    assn_test_ped = km.predict(test_data_.loc[:, ['latent_' + str(i) for i in range(latent_dim)]])
    centers_test_ped = pd.DataFrame(pd.Series(assn_test_ped).apply(lambda x: km.cluster_centers_[x]).tolist())
    predictions_kbn_test = pd.DataFrame([i.detach().cpu().numpy().reshape(-1) for i in
                                         model.decoding(torch.Tensor(centers_test_ped.to_numpy()).to(device))], index=vars_3).transpose()

    # # ideal R at nk_max
    # print(pd.Series([pearsonr(train_data_.loc[:, i], predictions_kbn.loc[:, i])[0]
    #                  for i in vars_3], index=vars_3).mean())
    # print(pd.Series([pearsonr(test_data_.loc[:, i], predictions_kbn_test.loc[:, i])[0]
    #                  for i in vars_3], index=vars_3).mean())

    ## distance matrix between centroids
    dist_matrix = cdist(km.cluster_centers_, km.cluster_centers_, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.nan)
    dist_matrix = pd.DataFrame(dist_matrix, index=range(nk_max), columns=range(nk_max))

    # ideal correlations
    cor_dkbn = pd.Series([pearsonr(train_data_.loc[:, i], predictions_kbn.loc[:, i])[0] for i in vars_3], index=vars_3)

    n_group = nk_max
    cor_final_ped = {}
    merge_map_ped = {}
    ped_km_clustering = {}
    km.cluster_centers_ = pd.DataFrame(km.cluster_centers_, index=range(nk_max))

    while n_group > nk_min:
        # get correlation for some
        if n_group in central_k:
            km_copy = copy.deepcopy(km)
            ped_km_clustering[n_group] = km_copy
            new_data_m = pd.DataFrame(pd.Series(km.labels_).apply(lambda x: (km.cluster_centers_).loc[x, ]))
            predictions_kbn = pd.DataFrame([i.detach().cpu().numpy().reshape(-1) for i in
                                            model.decoding(torch.Tensor(new_data_m.to_numpy()).to(device))], index=vars_3).transpose()
            cor_final_ped[n_group] = np.mean(list(vectorized_r(torch.Tensor(train_data_[vars_3].to_numpy()), torch.Tensor(predictions_kbn[vars_3].to_numpy()))))
            # pd.Series([pearsonr(train_data_.loc[:, i], predictions_kbn.loc[:, i])[0]
            #                                     for i in vars_3], index=vars_3)

        # perform iterative reduction
        decrease = Parallel(n_jobs=30)(delayed(process_y)(y, dist_matrix, n_group, km, train_data_,
                                                         latent_dim, model, device, vars_3) for y in range(30))
        decrease = [i for i in decrease if i != 9999]

        sorted_dist_matrix = np.sort(dist_matrix.to_numpy()[np.triu_indices(n=n_group, k=1)])
        to_merge_m = (dist_matrix == sorted_dist_matrix[np.argmin(decrease)])
        to_merge_m = to_merge_m[to_merge_m].stack().index.values[0]
        merge_map_ped[n_group] = to_merge_m

        km.cluster_centers_.loc[to_merge_m[0], :] = train_data_.loc[:,
                                            ['latent_' + str(i) for i in range(latent_dim)]].loc[np.isin(km.labels_, to_merge_m)].mean().values
        km.cluster_centers_.loc[to_merge_m[1], :] = train_data_.loc[:,
                                            ['latent_' + str(i) for i in range(latent_dim)]].loc[np.isin(km.labels_, to_merge_m)].mean().values
        km.cluster_centers_ = (km.cluster_centers_).drop_duplicates(keep='first')
        km.labels_[km.labels_ == np.max(to_merge_m)] = np.min(to_merge_m)

        new_data = pd.DataFrame(pd.Series(km.labels_).apply(lambda x: (km.cluster_centers_).loc[x, ]))
        n_group = km.cluster_centers_.shape[0]

        dist_matrix = cdist(km.cluster_centers_, km.cluster_centers_, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.nan)
        dist_matrix = pd.DataFrame(dist_matrix, index=km.cluster_centers_.index.values, columns=km.cluster_centers_.index.values)

    return km_reserve, ped_km_clustering, merge_map_ped, cor_final_ped, cor_dkbn


def get_test_predictions(test_data_, km_reserve, ped_km_clustering, merge_map_ped, model, device, vars_3, latent_dim, central_k, nk_max):
    n_clusters_list = central_k
    cor_final_c_test = {}
    pred_final_c_test = {i: None for i in n_clusters_list}
    predictions_central = {i: None for i in n_clusters_list}

    for n_clusters in n_clusters_list:
        cluster_merged_ped = {}
        for i in reversed(range((n_clusters + 1), (nk_max + 1))):
            k = min(merge_map_ped[i])
            cluster_merged_ped_list = [value for values in cluster_merged_ped.values() for value in values]
            keys = list(cluster_merged_ped.keys())
            if all([i in cluster_merged_ped_list for i in list(merge_map_ped[i])]):
                loc1 = keys[np.where([(k in cluster_merged_ped[x]) for x in keys])[0][0]]
                loc2 = keys[np.where([(max(merge_map_ped[i]) in cluster_merged_ped[x]) for x in keys])[0][0]]
                cluster_merged_ped[loc1] = list(set(cluster_merged_ped[loc1] + cluster_merged_ped[loc2]))
                cluster_merged_ped.pop(loc2)
            elif any([i in cluster_merged_ped_list for i in list(merge_map_ped[i])]):
                merging_cl = list(merge_map_ped[i])[np.where([i in cluster_merged_ped_list for i in list(merge_map_ped[i])])[0][0]]
                loc = keys[np.where([(merging_cl in cluster_merged_ped[x]) for x in keys])[0][0]]
                cluster_merged_ped[loc] = list(set(cluster_merged_ped[loc] + list(merge_map_ped[i])))
            else:
                cluster_merged_ped[k] = list(merge_map_ped[i])

        # get list of all clusters up to n_clusters
        all_nclus = list(set([value for values in [merge_map_ped[i] for i in range(min(merge_map_ped.keys()), (n_clusters + 1))] for value in values]))
        cluster_merged_ped_list = [value for values in cluster_merged_ped.values() for value in values]
        single_cluster = [i for i in all_nclus if i not in cluster_merged_ped_list]

        # add single cluster to the merge list
        for i in single_cluster:
            cluster_merged_ped[i] = [i]

        # use min value as name for each
        original_dict = cluster_merged_ped.keys()
        new_key_mapping = {i: min(cluster_merged_ped[i]) for i in list(cluster_merged_ped.keys())}
        cluster_merged_ped = {new_key_mapping.get(old_key, old_key): value for old_key, value in cluster_merged_ped.items()}
        cluster_merged_ped = dict(sorted(cluster_merged_ped.items()))

        # get prediction of test set
        predicted_clusters = km_reserve.predict(test_data_.loc[:, ['latent_' + str(i) for i in range(latent_dim)]])
        predicted_clusters_reduced = pd.Series(predicted_clusters).apply(lambda x: list(cluster_merged_ped.keys())[
            np.where([x in cluster_merged_ped[i] for i in cluster_merged_ped.keys()])[0][0]])
        predicted_centers_reduced = predicted_clusters_reduced.apply(lambda x: ped_km_clustering[n_clusters].cluster_centers_.loc[x, :])
        predictions_kbn_test = pd.DataFrame([i.detach().cpu().numpy().reshape(-1) for i in
                                             model.decoding(torch.Tensor(predicted_centers_reduced.to_numpy()).to(device))], index=vars_3).transpose()

        cor_final_c_test[n_clusters] = pd.Series(list(vectorized_r(torch.Tensor(test_data_[vars_3].to_numpy()), torch.Tensor(predictions_kbn_test[vars_3].to_numpy())))).mean() #pd.Series([pearsonr(test_data_.loc[:, i], predictions_kbn_test.loc[:, i])[0] for i in vars_3], index=vars_3)
        pred_final_c_test[n_clusters] = predictions_kbn_test

    return cor_final_c_test, pred_final_c_test


def main():
    # data = pd.read_csv(path + 'data/5-Final_allColumns_082024.csv')
    data = pd.read_csv(path + 'Python/paper_github/mock_data/mock_data.csv')
    data = preprocess_data(data)

    # Create masks
    data_test = data.copy()
    ndays = 730
    cv = ['3', '4']
    train_mrn, val_mrn, test_mrn = create_masks(data_test, ndays, cv, path)

    X_ = data_test.loc[:, data_test.columns.isin(input_var)].copy()
    y_ = data_test.loc[:, vars_3].copy()
    X_train, y_train = X_.loc[data_test.MedicalRecordNum.isin(train_mrn), :], y_.loc[data_test.MedicalRecordNum.isin(train_mrn), :]
    X_val, y_val = X_.loc[data_test.MedicalRecordNum.isin(val_mrn), :], y_.loc[data_test.MedicalRecordNum.isin(val_mrn), :]
    X_test, y_test = X_.loc[data_test.MedicalRecordNum.isin(test_mrn), :], y_.loc[data_test.MedicalRecordNum.isin(test_mrn), :]

    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train, X_val, X_test, y_train, y_val, y_test = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train, X_val, X_test, y_train, y_val, y_test = convert_to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, device=device)

    # Call the main training loop
    latent_dim = 16
    model = main_training_loop(X_train, y_train, X_val, y_val, latent_dim, device)

    # Get encoded latent representations
    latent_train, latent_val, latent_test = get_latent_rep(model, X_train, X_val, X_test)

    # Map columns
    data_test = map_columns(data_test)

    # Create latent representation dataframes
    train_data, val_data, test_data, all_data = create_df_with_latent(data_test, latent_train, latent_val, latent_test, y_train, y_val, y_test, train_mrn, val_mrn, test_mrn, latent_dim, vars_3)

    # doing central line of NEONATAL protocol as an example
    central_k = [2, 5, 10, 20] #[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 50, 80, 90]
    train_data_ = train_data.loc[((train_data.ProtocolName == 'NEONATAL') & (train_data.LineID==1)), :]
    test_data_ = test_data.loc[((test_data.ProtocolName == 'NEONATAL') & (test_data.LineID==1)), :]

    nk_max = 21# originally at but reduced to 20 to suit mock data 91
    nk_min = 1
    # Perform clustering
    km_reserve, ped_km_clustering, merge_map_ped, cor_final_ped, cor_dkbn = perform_clustering(train_data_, test_data_, latent_dim, model, device, vars_3, central_k, nk_max, nk_min)

    # Get test predictions
    cor_final_c_test, pred_final_c_test = get_test_predictions(test_data_, km_reserve, ped_km_clustering, merge_map_ped, model, device, vars_3, latent_dim, central_k, nk_max)

    print(cor_final_c_test)

if __name__ == "__main__":
    main()
