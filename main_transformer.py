import os
os.chdir("/home/tpjoe/tpjoe@stanford.edu/project_TPN/Python/paper_github")
path = "/home/tpjoe/tpjoe@stanford.edu/project_TPN/"
import pandas as pd
import numpy as np
import gc
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import copy
from itertools import compress
from sklearn.cluster import KMeans

import importlib
import model
import train
import utils
importlib.reload(model)
importlib.reload(train)
importlib.reload(utils)

from model import *
from train import *
from utils import *
import time


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
meta_var = ['AANDC_code', 'AdultMVIDose', 'Osm']
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


def create_masks(data_test, ndays, cv):
    # mrn_randomized = pd.read_csv(f"/home/tpjoe/tpjoe@stanford.edu/project_TPN/data/cv_fold_id_{ndays}day.csv")
    mrn_randomized = pd.read_csv(path + 'Python/paper_github/mock_data/mock_TrainTest.csv')
    mrn_randomized['ttv'] = mrn_randomized[cv].apply(lambda x: 'Test' if ((x[cv[0]]=='Test') | (x[cv[1]]=='Test')) else \
                                                                 'Val' if ((x[cv[0]]=='Val')) else \
                                                                 'Train', axis=1)
    mrn_cv = data_test.reset_index().merge(mrn_randomized, left_on='MedicalRecordNum', right_on='shuffled_mrn')
    masks = mrn_cv.drop_duplicates('sMRN')[['ttv']]
    mask_train = (masks.ttv=='Train').tolist()
    mask_val = (masks.ttv=='Val').tolist()
    mask_test = (masks.ttv=='Test').tolist()
    return mask_train, mask_val, mask_test


def transform_sequence_data(X_list, y_list, meta_list):
    PAD_IDX, BOS_IDX, EOS_IDX, NAN_IDX = 9999, 2, 3, -15
    X_list = [torch.nan_to_num(i, nan=NAN_IDX) for i in X_list]
    y_list = [torch.nan_to_num(i, nan=NAN_IDX) for i in y_list]
    meta_list = [torch.nan_to_num(i, nan=NAN_IDX) for i in meta_list]
    y_SOS = torch.Tensor([BOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
    y_EOS = torch.Tensor([EOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
    meta_SOS = torch.Tensor([BOS_IDX]).repeat(meta_list[0].shape[1]).unsqueeze(0)
    meta_EOS = torch.Tensor([EOS_IDX]).repeat(meta_list[0].shape[1]).unsqueeze(0)
    X_EOS = torch.Tensor([EOS_IDX]).repeat(X_list[0].shape[1]).unsqueeze(0)
    X_list = [torch.vstack((x, X_EOS)) for x in X_list]
    y_list = [torch.vstack((y_SOS, y, y_EOS)) for y in y_list]
    meta_list =[torch.vstack((meta_SOS, m, meta_EOS)) for m in meta_list]
    return X_list,y_list,meta_list,PAD_IDX,BOS_IDX,EOS_IDX


def scale_data(X_list, y_list, mask_train):
    input_scaler = StandardScaler()
    input_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape) == 1 else i for i in list(compress(X_list, mask_train))]))
    
    output_scaler = StandardScaler()
    output_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape) == 1 else i for i in list(compress(y_list, mask_train))]))
    
    X_list_scaled = [torch.Tensor(input_scaler.transform(i)) for i in X_list]
    y_list_scaled = [torch.Tensor(output_scaler.transform(i)) for i in y_list]
    
    return X_list_scaled, y_list_scaled, input_scaler, output_scaler


def define_model_and_parameters(y_shape, X_shape, device=device):
    DEVICE = device
    torch.manual_seed(0)
    TGT_N_TASKS = y_shape[1]
    SRC_SIZE = X_shape[1]
    TGT_SIZE = y_shape[1]
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, TGT_N_TASKS, SRC_SIZE, TGT_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    return transformer, optimizer, DEVICE, BATCH_SIZE


def pretrain_model(transformer, optimizer, train_dataloader, val_dataloader, PAD_IDX, BOS_IDX, EOS_IDX, DEVICE, num_epochs=10000, patience=5, delta=0.001):
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    best_val_loss = 9999
    for epoch in range(1, num_epochs + 1):
        train_loss, r_train = train_teacherEnforce(transformer, optimizer, train_dataloader, PAD_IDX, BOS_IDX, EOS_IDX, DEVICE)
        val_loss, r_val = evaluate(transformer, val_dataloader, PAD_IDX, BOS_IDX, EOS_IDX, DEVICE)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},\
                 Train r: {r_train:.3f}, Val r: {r_val:.3f}"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = transformer.state_dict()
        if early_stopping(val_loss):
            print("Early stopping!")
            break
    
    transformer.load_state_dict(best_model_state)
    return transformer


def train_model(transformer, optimizer, train_dataloader, val_dataloader, y_shape, PAD_IDX, BOS_IDX, DEVICE, vars_3, num_epochs=5000, patience=10, delta=0.0001):
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss, y_all_, tgt_all_, n_samples_ = train_autoRegressive_sl_AI(transformer, y_shape[1], train_dataloader,
                                                                              optimizer, PAD_IDX, BOS_IDX, DEVICE, sl=60)
        val_loss, y_all_val_, tgt_all_val_, n_samples_ = inference_AI(transformer, y_shape[1], val_dataloader, 
                                                                      optimizer, PAD_IDX, BOS_IDX, DEVICE)
        
        r_train = np.mean(list(vectorized_r( np.concatenate(y_all_).reshape(-1, len(vars_3)), np.concatenate(tgt_all_).reshape(-1, len(vars_3)) )))
        r_val = np.mean(list(vectorized_r( np.concatenate(y_all_val_).reshape(-1, len(vars_3)), np.concatenate(tgt_all_val_).reshape(-1, len(vars_3)) )))
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},\
                Train r: {r_train:.3f}, Val r: {r_val:.3f}"))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = transformer.state_dict()
        if early_stopping(val_loss):
            print("Early stopping!")
            break
    
    transformer.load_state_dict(best_model_state)
    return transformer


def train_model_PI(transformer, optimizer, train_dataloader, val_dataloader, y_shape, PAD_IDX, BOS_IDX, DEVICE, vars_3, 
                   train_meta_dataloader, input_scaler, output_scaler, num_epochs=5000, patience=10, delta=0.0001):
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_loss, y_all_, tgt_all_, n_samples_ = train_autoRegressive_PI(transformer, y_shape[1], train_meta_dataloader, optimizer, PAD_IDX, BOS_IDX, 
                                                                           DEVICE, input_scaler, output_scaler, vars_3, input_var, sl=30, 
                                                                           inference=False, use_osm_loss=False)
        val_loss, y_all_val_, tgt_all_val_, n_samples_ = inference_AI(transformer, y_shape[1], val_dataloader, 
                                                                      optimizer, PAD_IDX, BOS_IDX, DEVICE)
        
        r_train = np.mean(list(vectorized_r( np.concatenate(y_all_).reshape(-1, len(vars_3)), np.concatenate(tgt_all_).reshape(-1, len(vars_3)) )))
        r_val = np.mean(list(vectorized_r( np.concatenate(y_all_val_).reshape(-1, len(vars_3)), np.concatenate(tgt_all_val_).reshape(-1, len(vars_3)) )))
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},\
                Train r: {r_train:.3f}, Val r: {r_val:.3f}"))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = transformer.state_dict()
        if early_stopping(val_loss):
            print("Early stopping!")
            break
    
    transformer.load_state_dict(best_model_state)
    return transformer


def cluster_and_train(all_k, y_all_, y_, vars_3, train_dataloader_sub, val_dataloader_sub, test_dataloader_sub, PAD_IDX, BOS_IDX, DEVICE, state_dict,
                      NUM_ENCODER_LAYERS=3, NUM_DECODER_LAYERS=3, EMB_SIZE=512, NHEAD=8, TGT_N_TASKS=17, SRC_SIZE=69, TGT_SIZE=17, FFN_HID_DIM=512):
    r_test = {}
    for n_clusters in all_k:
        print(n_clusters)
        tol = 0.15 if n_clusters < 50 else 0.2
        trans_IDEC = Seq2SeqTransformer_IDEC(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                             NHEAD, TGT_N_TASKS, SRC_SIZE, TGT_SIZE, FFN_HID_DIM,
                                             n_z=len(vars_3), n_clusters=n_clusters, device=DEVICE)
        trans_IDEC.transformer.load_state_dict(state_dict)
        trans_IDEC = trans_IDEC.to(DEVICE)
        optimizer = torch.optim.Adam(trans_IDEC.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        hidden = (np.concatenate(y_all_).reshape(-1, len(vars_3)))
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(hidden)
        clus_pred = kmeans.predict(hidden)
        clus_pred_last = clus_pred
        trans_IDEC.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(DEVICE)
        torch.cuda.empty_cache()
        gc.collect()
        for epoch in range(1000):
            if ((epoch == 0) | (epoch % 4 == 0)):
                train_loss, y_all_, tgt_all_, n_samples_, tmp_q = inference_q2_AI(trans_IDEC, y_.shape[1], train_dataloader_sub,
                                                                                  optimizer, PAD_IDX, BOS_IDX, DEVICE)
                p_train = target_distribution(torch.cat(tmp_q))
                clus_pred = torch.cat(tmp_q).numpy().argmax(1)
                val_loss, y_all_val_, tgt_all_val_, n_samples_, tmp_q = inference_q2_AI(trans_IDEC, y_.shape[1], val_dataloader_sub,
                                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE)
                p_val = target_distribution(torch.cat(tmp_q))
                delta_label = np.sum(clus_pred != clus_pred_last).astype(np.float32) / clus_pred.shape[0]
                clus_pred_last = clus_pred
                print('Difference = ', delta_label)
                if epoch > 0 and delta_label < tol:
                    print('delta_label {:.4f}'.format(delta_label), '< tol', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            train_loss, y_all_, tgt_all_, n_samples_, tmp_q = train_autoRegressive_q2_AI(trans_IDEC.cluster_layer.data, p_train,
                                                                                         trans_IDEC, y_.shape[1], train_dataloader_sub,
                                                                                         optimizer, PAD_IDX, BOS_IDX, DEVICE)
            clus_pred = torch.Tensor(np.concatenate((tmp_q))).numpy().argmax(1)
            pred_train = torch.vstack([trans_IDEC.cluster_layer.data[i, :] for i in clus_pred]).cpu().numpy()
            val_loss, y_all_val_, tgt_all_val_, _, tmp_q = inference_q2_AI(trans_IDEC, y_.shape[1], val_dataloader_sub,
                                                                           optimizer, PAD_IDX, BOS_IDX, DEVICE)
            clus_pred = torch.cat(tmp_q).numpy().argmax(1)
            pred_val = torch.vstack([trans_IDEC.cluster_layer.data[i, :] for i in clus_pred]).cpu().numpy()
            r_train = np.mean(list(vectorized_r( pred_train, np.concatenate(tgt_all_).reshape(-1, len(vars_3)) )))
            r_val = np.mean(list(vectorized_r( pred_val, np.concatenate(tgt_all_val_).reshape(-1, len(vars_3)) )))
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, \
            Train r: {np.mean(r_train):.3f}, Val r: {np.mean(r_val):.3f}"))
        test_loss, y_all_test_, tgt_all_test_, _, tmp_q = inference_q2_AI(trans_IDEC, y_.shape[1], test_dataloader_sub,
                                                                          optimizer, PAD_IDX, BOS_IDX, DEVICE)
        clus_pred = torch.cat(tmp_q).numpy().argmax(1)
        pred_test = torch.vstack([trans_IDEC.cluster_layer.data[i, :] for i in clus_pred]).cpu().numpy()
        r_test[n_clusters] = [np.mean(list(vectorized_r( pred_test, np.concatenate(tgt_all_test_).reshape(-1, len(vars_3)) )))]
    return r_test


def main(PI):
    # Load data
    # data = pd.read_csv(path + 'data/5-Final_allColumns_082024.csv')
    device = 'cuda:0'
    data = pd.read_csv(path + 'Python/paper_github/mock_data/mock_data.csv')

    # Preprocess data
    data = preprocess_data(data)

    # Prepare data
    data_test = data.copy()
    X_list, y_list, meta_list, X_, y_, meta_ = prepare_data(data_test, input_var, vars_3, meta_var)

    # Create masks
    ndays = 730
    cv = ['3', '4']
    mask_train, mask_val, mask_test = create_masks(data_test, ndays, cv)

    # Scale data
    X_list, y_list, input_scaler, output_scaler = scale_data(X_list, y_list, mask_train)
    X_list, y_list, meta_list, PAD_IDX, BOS_IDX, EOS_IDX = transform_sequence_data(X_list, y_list, meta_list)

    # Define model and parameters
    transformer, optimizer, DEVICE, BATCH_SIZE = define_model_and_parameters(y_.shape, X_.shape, device=device)

    # Create dataloaders
    train_meta_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_train)),
                                                                list(compress(y_list, mask_train)),
                                                                meta_data=list(compress(meta_list, mask_train)),
                                                                batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
    train_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_train)),
                                                            list(compress(y_list, mask_train)),
                                                            batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
    val_meta_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_val)),
                                                            list(compress(y_list, mask_val)),
                                                            meta_data=list(compress(meta_list, mask_val)),
                                                            batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
    val_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_val)),
                                                        list(compress(y_list, mask_val)),
                                                        batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
    test_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_test)),
                                                        list(compress(y_list, mask_test)),
                                                        batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)

    # Pretrain model
    transformer = pretrain_model(transformer, optimizer, train_dataloader, val_dataloader, PAD_IDX, BOS_IDX, EOS_IDX, DEVICE)

    # Evaluate pretrained model on test set
    test_results = {}
    test_loss, y_all_test_, tgt_all_test_, _ = inference_AI(transformer, y_.shape[1], test_dataloader,
                                                            optimizer, PAD_IDX, BOS_IDX, DEVICE)
    gt = np.concatenate(tgt_all_test_).reshape(-1, len(vars_3))
    preds = np.concatenate(y_all_test_).reshape(-1, len(vars_3))
    test_results['mean_r_pretrain'] = pd.Series(list(vectorized_r(gt, preds))).mean()

    # Train model
    if PI == False:
        transformer = train_model(transformer, optimizer, train_dataloader, val_dataloader, y_.shape, PAD_IDX, BOS_IDX, DEVICE, vars_3)
    else:
        transformer = train_model_PI(transformer, optimizer, train_dataloader, val_dataloader, y_.shape, PAD_IDX, BOS_IDX, DEVICE, vars_3, 
                                     train_meta_dataloader, input_scaler, output_scaler)

    # Evaluate trained model on test set
    test_loss, y_all_test_, tgt_all_test_, _ = inference_AI(transformer, y_.shape[1], test_dataloader,
                                                            optimizer, PAD_IDX, BOS_IDX, DEVICE)
    gt = np.concatenate(tgt_all_test_).reshape(-1, len(vars_3))
    preds = np.concatenate(y_all_test_).reshape(-1, len(vars_3))
    test_results['mean_r_posttrain'] = pd.Series(list(vectorized_r(gt, preds))).mean()

    # start clustering using all lines as an example
    protocol = 'All'
    if protocol=='Central':
        condition = (data_test.ProtocolName_NEONATAL.isin([1])) & (data_test.LineID_2.isin([0]))
    elif protocol=='Peri':
        condition = ((data_test.ProtocolName_NEONATAL.isin([1])) & (data_test.LineID_2==1))
    elif protocol=='Pedi':
        condition = (data_test.ProtocolName_NEONATAL.isin([0]))
    else:
        condition = (data_test.ProtocolName_NEONATAL.isin([0, 1]))
    

    data_test_sub = data_test.loc[condition, :].copy()
    X_list, y_list, meta_list, X_, y_, _ = prepare_data(data_test_sub, input_var, vars_3, meta_var)

    # Create masks
    ndays = 730
    cv = ['3', '4']
    mask_train, mask_val, mask_test = create_masks(data_test_sub, ndays, cv)

    # Scale data
    X_list, y_list, input_scaler, output_scaler = scale_data(X_list, y_list, mask_train)
    X_list, y_list, meta_list, PAD_IDX, BOS_IDX, EOS_IDX = transform_sequence_data(X_list, y_list, meta_list)

    # Define model and parameters
    transformer, optimizer, DEVICE, BATCH_SIZE = define_model_and_parameters(y_.shape, X_.shape, device=device)

    # Create dataloaders
    train_dataloader_sub = create_variable_length_dataloader(list(compress(X_list, mask_train)),
                                                            list(compress(y_list, mask_train)),
                                                            batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
    val_dataloader_sub = create_variable_length_dataloader(list(compress(X_list, mask_val)),
                                                        list(compress(y_list, mask_val)),
                                                        batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
    test_dataloader_sub = create_variable_length_dataloader(list(compress(X_list, mask_test)),
                                                        list(compress(y_list, mask_test)),
                                                        batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)

    # Call the function
    all_k = [5, 10, 15]  # Example cluster sizes
    loss, y_all_, tgt_all_, _ = inference_AI(transformer, y_.shape[1], train_dataloader_sub, 
                                                                    "_", PAD_IDX, BOS_IDX, DEVICE)
    r_test = cluster_and_train(all_k, y_all_, y_, vars_3, train_dataloader_sub, val_dataloader_sub, test_dataloader_sub, PAD_IDX, BOS_IDX, DEVICE, transformer.state_dict())
    test_results['cluster_r'] = (r_test)
    
    print(test_results)


if __name__ == "__main__":
    PI = True
    main(PI)
