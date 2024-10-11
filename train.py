from utils import *
from itertools import compress
import gc
import numpy as np
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans as th_Kmeans
import torch
import math
from tqdm import tqdm
import contextlib
from utils import *

def train_teacherEnforce(model, optimizer, train_dataloader, PAD_IDX,  BOS_IDX, EOS_IDX, DEVICE):
    model.train()
    losses = 0
    r = 0

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :, :].to(DEVICE)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, DEVICE)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE), \
                                                                 src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask).to(DEVICE)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :, :].to(DEVICE)
        loss = mse_loss(logits.reshape(-1), tgt_out.reshape(-1), 
                        ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE), reduction='mean')
        r += pearsonr_corr(logits.reshape(-1), tgt_out.reshape(-1), ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader)), r / len(list(train_dataloader))


def evaluate(model, val_dataloader, PAD_IDX,  BOS_IDX, EOS_IDX, DEVICE):
    # this function test the performance based on teacher forcing
    model.eval()
    losses = 0
    r = 0
    with torch.no_grad():
        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:-1, :, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, DEVICE)
            
            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :, :]
            loss = mse_loss(logits.reshape(-1), tgt_out.reshape(-1), 
                            ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE), reduction='mean')
            r += pearsonr_corr(logits.reshape(-1), tgt_out.reshape(-1), ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE))
            losses += loss.item()

    return losses / len(list(val_dataloader)), r / len(list(val_dataloader))


def run_batch3_AI(n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, cutoff, PAD_IDX=9999, sl=30):
    # this function runs part of the inference for each sample
    max_len = max(max_lens)
    batch_size = src.shape[1]
    ys = torch.ones(1, batch_size, n_tasks).fill_(BOS_IDX).to(DEVICE)
    intervene = torch.ones(1, batch_size).to(DEVICE)
    # intervene = [[]]*batch_size
    for i in range(max_len-1):
        ys_sl = ys if ys.shape[0] <= sl else ys[-sl:, :, :]
        src_sl = src[:(i+1), :, :]
        src_pad_mask = (src_sl==PAD_IDX).all(axis=2).t()
        tgt_pad_mask = src_pad_mask
        src_mask = torch.zeros((src_sl.shape[0], src_sl.shape[0]), device=DEVICE).type(torch.bool)
        tgt_mask = torch.zeros((ys_sl.shape[0], ys_sl.shape[0]), device=DEVICE).type(torch.bool)
        memory = transformer.encode(src_sl, src_mask, src_key_padding_mask=src_pad_mask)
        out = transformer.decode(ys_sl, memory, tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = out.transpose(0, 1)
        pred = transformer.generator(out[:, -1, :])
        pred = pred.reshape([-1, pred.shape[0], pred.shape[1]])
        tgt_fill = tgt[(i+1), :, :]
        tgt_fill = tgt_fill.reshape([-1, tgt_fill.shape[0], tgt_fill.shape[1]])
        # dist = np.abs(torch.mean(pred - tgt_fill).detach().cpu().numpy())
        dist = torch.norm(pred - tgt_fill, dim=-1)
        # intervene = [x + [(dist[0][i].detach().cpu().flatten().numpy()[0] > cutoff).astype(int)] for i, x in enumerate(intervene)]
        # intervene.append(intervene_batch.cpu().numpy())
        yss = []
        for ii, d in enumerate(dist[0]):
            if d> cutoff:
                yss.append(tgt_fill[:, ii, :])
            else:
                yss.append(pred[:, ii, :])
        ys = torch.cat([ys, torch.concat(yss, dim=0).unsqueeze(0)], axis=0)
        # ys = torch.cat([ys, torch.where(dist > cutoff, tgt_fill.reshape(1, n_tasks, -1), pred.reshape(1, n_tasks, -1)).reshape(1, -1, n_tasks)], dim=0)
        intervene = torch.cat([intervene, torch.where(dist > cutoff, torch.Tensor([1]).to(DEVICE), torch.Tensor([0]).to(DEVICE))], dim=0)

    ys_ = [ys[1:max_lens[sample_n], sample_n:(sample_n+1), :] for sample_n in range(batch_size)]
    tgt_ = [tgt[1:max_lens[sample_n], sample_n:(sample_n+1), :] for sample_n in range(batch_size)]
    intervene = [intervene[1:max_lens[sample_n], sample_n:(sample_n+1)].flatten().detach().cpu().numpy().tolist() for sample_n in range(batch_size)]
    ys = torch.concat(ys_)
    ys_noBOSEOS = [i.detach().cpu().numpy() for i in ys_]
    tgt = torch.concat(tgt_)
    tgt_noBOSEOS = [i.detach().cpu().numpy() for i in tgt_]
    # intervene = [intervene[sample_n][:(max_lens[sample_n]-1)] for sample_n in range(batch_size)]
    torch.cuda.empty_cache()
    gc.collect()
    
    return ys, tgt, ys_noBOSEOS, tgt_noBOSEOS, intervene



def run_batch_sliding_AI(n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, PAD_IDX=9999, sl=30):
    max_len = max(max_lens)
    batch_size = src.shape[1]
    ys = torch.ones(1, batch_size, n_tasks).fill_(BOS_IDX).to(DEVICE)
    all_preds = []
    for i in range(max_len - 1):
        ys_sl = ys if ys.shape[0] <= sl else ys[-sl:, :, :]
        src_sl = src[:(i+1), :, :]
        src_sl = src_sl if src_sl.shape[0] <= sl else src_sl[-sl:, :, :]
        
        src_pad_mask = (src_sl==PAD_IDX).all(axis=2).t()
        tgt_pad_mask = src_pad_mask
        
        src_mask = torch.zeros((src_sl.shape[0], src_sl.shape[0]), device=DEVICE).type(torch.bool)
        tgt_mask = torch.zeros((ys_sl.shape[0], ys_sl.shape[0]), device=DEVICE).type(torch.bool)
        
        memory = transformer.encode(src_sl, src_mask, src_key_padding_mask=src_pad_mask)
        
        out = transformer.decode(ys_sl, memory, tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = out.transpose(0, 1)
        pred = transformer.generator(out[:, -1, :])
        pred = pred.reshape([-1, pred.shape[0], pred.shape[1]])
        
        ys = torch.cat([ys, pred], dim=0)
        all_preds.append(pred.detach().cpu().numpy())

    ys_ = [ys[1:max_lens[sample_n], sample_n:(sample_n+1), :] for sample_n in range(batch_size)]
    tgt_ = [tgt[1:max_lens[sample_n], sample_n:(sample_n+1), :] for sample_n in range(batch_size)]
    ys = torch.concat(ys_)
    ys_noBOSEOS = [i.detach().cpu().numpy() for i in ys_]
    tgt = torch.concat(tgt_)
    tgt_noBOSEOS = [i.detach().cpu().numpy() for i in tgt_]
    torch.cuda.empty_cache()
    gc.collect()
    
    return ys, tgt, ys_noBOSEOS, tgt_noBOSEOS


def train_autoRegressive_q2_AI(centers, p, trans_IDEC, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, sl=30):
    trans_IDEC.train()
    trans_IDEC.transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    count = 0
    last_q = 0
    tmp_q = []
    progress_bar = tqdm(total=len(train_dataloader.dataset), desc='Training...')
    for i, (src, tgt) in enumerate(train_dataloader):
        count += 1
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
        res = run_batch_sliding_AI(n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC.transformer, src, tgt, PAD_IDX, sl)
        yss, tgts, y_all_batch, tgt_all_batch = res
        y_all.extend(y_all_batch)
        tgt_all.extend(tgt_all_batch)
        yss = yss.to(DEVICE) #torch.concat(yss, axis=0).to(DEVICE)
        tgts = tgts.to(DEVICE) #torch.concat(tgts, axis=0).to(DEVICE)
        z = torch.Tensor(np.concatenate(y_all_batch).reshape(-1, n_tasks)).to(DEVICE)
        q = trans_IDEC(z).detach().cpu()
        tmp_q += [q]
        optimizer.zero_grad()
        loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2).cpu() + F.kl_div(q.log(), p[last_q:(last_q+q.shape[0]), :])
        loss = loss.to(DEVICE)
        loss.backward(retain_graph=True)
        optimizer.step()
        losses += loss.item()
        n_samples += src.shape[1]
        progress_bar.update(src.shape[1])
        last_q += q.shape[0]
        torch.cuda.empty_cache()
        gc.collect()
    
    return losses/count, y_all, tgt_all, n_samples, tmp_q



def inference_prescription_AI(transformer, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, cutoff, sl=30):
    # this function will run inference such that the top 10% furthest prescription vs. AI are replaced with
    # the prescription itself.
    transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    intervene = []
    count = 0
    progress_bar = tqdm(total=len(dataloader.dataset), desc='Training...')
    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            res = run_batch3_AI(n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, cutoff, PAD_IDX, sl)
            yss, tgts, y_all_batch, tgt_all_batch, intervene_ = res
            y_all.extend(y_all_batch)
            tgt_all.extend(tgt_all_batch)
            intervene.extend(intervene_)
            # memories.extend(memories_batch)
            loss = torch.mean((yss.reshape(-1) - tgts.reshape(-1)) ** 2)
            losses += loss.item()
            n_samples += src.shape[1]
            progress_bar.update(src.shape[1])
            count += 1
            torch.cuda.empty_cache()
            gc.collect()
            
    return losses / len(list(dataloader)), y_all, tgt_all, n_samples, intervene


def inference_q2_AI(trans_IDEC, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, sl=300):
    trans_IDEC.transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    tmp_q = []
    count = 0
    progress_bar = tqdm(total=len(dataloader.dataset), desc='Training...')
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            res = run_batch_sliding_AI(n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC.transformer, src, tgt, PAD_IDX, sl)
            yss, tgts, y_all_batch, tgt_all_batch = res
            y_all.extend(y_all_batch)
            tgt_all.extend(tgt_all_batch)
            # memories.extend(memories_batch)
            loss = torch.mean((yss.reshape(-1) - tgts.reshape(-1)) ** 2)
            losses += loss.item()
            n_samples += src.shape[1]
            progress_bar.update(src.shape[1])
            count += 1
            torch.cuda.empty_cache()
            gc.collect()

    z = torch.Tensor(np.concatenate(y_all).reshape(-1, n_tasks)).to(DEVICE)
    tmp_q += [trans_IDEC(z).detach().cpu()]
    
    return losses / len(list(dataloader)), y_all, tgt_all, n_samples, tmp_q



def train_autoRegressive_sl_AI(transformer, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, sl=30):
    transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0
    progress_bar = tqdm(total=len(train_dataloader.dataset), desc='Training...')
    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
        res = run_batch_sliding_AI(n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, PAD_IDX, sl)
        yss, tgts, y_all_batch, tgt_all_batch = res
        
        y_all.extend(y_all_batch)
        tgt_all.extend(tgt_all_batch)
        # memories.extend(memories_batch)
        
        optimizer.zero_grad()
        loss = torch.mean((yss.reshape(-1) - tgts.reshape(-1)) ** 2)
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        n_samples += src.shape[1]
        progress_bar.update(src.shape[1])
        count += 1
        torch.cuda.empty_cache()
        gc.collect()

    return losses / count, y_all, tgt_all, n_samples#, memories


def train_autoRegressive_PI(transformer, n_tasks, train_meta_dataloader, optimizer, PAD_IDX, BOS_IDX, 
                            DEVICE, input_scaler, output_scaler, vars_3, input_var, sl=30, 
                            inference=False, use_osm_loss=False):
    if inference:
        transformer.eval()
    else:
        transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0
    n_tasks = len(vars_3)
    progress_bar = tqdm(total=len(train_meta_dataloader.dataset), desc='Training...')
    # 
    with torch.no_grad() if inference else contextlib.nullcontext():
        for src, tgt, meta in train_meta_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            meta = meta.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            meta_all = torch.concat([meta[1:max_lens[sample_n], sample_n:(sample_n+1), :] for sample_n in range(src.shape[1])])
            src_all = torch.concat([src[:(max_lens[sample_n]-1), sample_n:(sample_n+1), :] for sample_n in range(src.shape[1])])
            res = run_batch_sliding_AI(n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, PAD_IDX, sl)
            yss, tgts, y_all_batch, tgt_all_batch = res

            y_all.extend(y_all_batch)
            tgt_all.extend(tgt_all_batch)
            # memories.extend(memories_batch)

            optimizer.zero_grad()
            # reverse transform
            scale_out = torch.tensor(output_scaler.scale_, device=yss.device).view(1, 1, -1)
            mean_out = torch.tensor(output_scaler.mean_, device=yss.device).view(1, 1, -1)
            rev_pred = yss * scale_out + mean_out
            scale_in = torch.tensor(input_scaler.scale_, device=yss.device).view(1, 1, -1)
            mean_in = torch.tensor(input_scaler.mean_, device=yss.device).view(1, 1, -1)
            rev_src = src_all * scale_in + mean_in

            # define individual input variables
            indices = get_variable_indices(input_var)
            weight = rev_src[:, :, indices['TodaysWeight']]
            vtbi = rev_src[:, :,indices['VTBI']]
            peri = rev_src[:, :, indices['LineID_2']].round()==1
            omegaven = rev_src[:, :, indices['FatProduct_Omegaven_10']].round()
            smof = rev_src[:, :, indices['FatProduct_SMOFlipid_20']].round()
            AdultMVIDose = meta_all[:, :, 1]

            # define individual output variables
            indices = get_variable_indices(vars_3)
            dd_pred = rev_pred[:, :, indices['DexDose']]
            fat_pred = rev_pred[:, :, indices['FatDose']]
            act_pred = rev_pred[:, :, indices['Acetate']]
            ca_pred = rev_pred[:, :, indices['Calcium']]
            mvi_pred = rev_pred[:, :, indices['MVIDose']]
            aa_pred = rev_pred[:, :,indices['AADose']]
            na_pred = rev_pred[:, :, indices['Sodium']]
            k_pred = rev_pred[:, :, indices['Potassium']]
            po4_pred = rev_pred[:, :, indices['Phosphate']]
            cl_pred = rev_pred[:, :, indices['Chloride']]
            zn_pred = rev_pred[:, :, indices['Zinc']]
            mg_pred = rev_pred[:, :, indices['Magnesium']]
            levocar_pred = rev_pred[:, :, indices['Levocar']]

            # get predicted values
            cl_pred_supposed = vectorized_cal_chloride(na_pred, k_pred, act_pred, aa_pred, po4_pred, meta_all[:, :, 0])
            osm_pred = vectorized_get_osm(weight, vtbi, smof, omegaven, AdultMVIDose, 
                            aa_pred, dd_pred, fat_pred, mvi_pred, 
                            po4_pred, na_pred, k_pred, ca_pred,
                            zn_pred, mg_pred, levocar_pred)
            soln_factor = ((ca_pred/430.373*weight*2/vtbi*1000)**0.863 * (po4_pred*weight/vtbi*1000)**1.19)/(aa_pred*weight/vtbi*100)
            # ca_mg = ca_pred * weight <<<<<<<<<

            # define loss
            aa_act_loss = torch.relu(0.96*aa_pred - act_pred)
            min_loss = torch.relu(((-0.01 - mean_out) / scale_out)-yss.reshape(-1, n_tasks))
            cl_loss = (((cl_pred-mean_out[:, :, indices['Chloride']])/scale_out[:, :, indices['Chloride']]) - \
                        ((cl_pred_supposed-mean_out[:, :, indices['Chloride']])/scale_out[:, :, indices['Chloride']])) ** 2
            osm_loss1 = torch.where(peri.any(), torch.relu(osm_pred[peri]-950)**2, torch.tensor(0.0, device=osm_pred.device))
            osm_loss2 = torch.where(peri.any(), torch.relu((meta_all[:, :, 2][peri]-50)-osm_pred)**2, torch.tensor(0.0, device=osm_pred.device))
            osm_loss = osm_loss1 + 1*osm_loss2
            osm_diff_loss = torch.where(peri.any(), ((osm_pred - 1252) / 344 - (meta_all[:, :, 2] - 1252) / 344)**2, torch.tensor(0.0, device=osm_pred.device))
            dd_loss = torch.where(peri.any(), torch.relu((dd_pred[peri]-mean_out[:, :, indices['DexDose']])/scale_out[:, :, indices['DexDose']]-\
                                                        (12.5-mean_out[:, :, indices['DexDose']])/scale_out[:, :, indices['DexDose']]), torch.tensor(0.0, device=dd_pred.device))
            # capo4_loss = torch.relu(soln_factor - 200) <<<<<<<<<<
            pred_loss = ((yss[:, :, 0:17].reshape(-1) - tgts[:, :, 0:17].reshape(-1)) ** 2)

            ca_limit = torch.where(peri, 3, 8)
            # ca_loss = ((yss[peri.flatten(), :, indices['Calcium']].reshape(-1) - tgts[peri.flatten(), :, indices['Calcium']].reshape(-1)) ** 2)
            # k_loss = ((yss[:, :, indices['Potassium']].reshape(-1) - tgts[:, :, indices['Potassium']].reshape(-1)) ** 2)
            # na_loss = ((yss[:, :, indices['Sodium']].reshape(-1) - tgts[:, :, indices['Sodium']].reshape(-1)) ** 2)
            ca_solbility_loss = torch.relu(ca_pred * weight/vtbi - ca_limit)

            loss = torch.mean(pred_loss) + torch.mean(aa_act_loss) + torch.mean(torch.sqrt(cl_loss)) + 17*torch.mean(min_loss)
            if use_osm_loss:
                loss = loss + torch.mean(dd_loss) + torch.mean(pred_loss) + 0.00001*torch.mean(osm_loss) + 4*torch.mean(torch.sqrt(cl_loss))# + torch.mean(osm_diff_loss) # #  0.0001*
                            # 0.0000001*torch.mean(osm_loss) + 10*torch.mean(min_loss) #0.0000005*torch.mean(osm_loss) # bottom limit 950
                # 0.000005*torch.mean(osm_loss) + 10*torch.mean(min_loss) #0.0000005*torch.mean(osm_loss) #top <- limit 1100

            if not inference:
                loss.backward()
                optimizer.step()
            
            losses += loss.item()
            n_samples += src.shape[1]
            progress_bar.update(src.shape[1])
            count += 1
            torch.cuda.empty_cache()
            gc.collect()

    return losses / count, y_all, tgt_all, n_samples#, memories


def inference_AI(transformer, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, sl=300):
    transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0
    progress_bar = tqdm(total=len(dataloader.dataset), desc='Training...')
    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            res = run_batch_sliding_AI(n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, PAD_IDX, sl)
            yss, tgts, y_all_batch, tgt_all_batch = res
            
            y_all.extend(y_all_batch)
            tgt_all.extend(tgt_all_batch)
            # memories.extend(memories_batch)
            loss = torch.mean((yss.reshape(-1) - tgts.reshape(-1)) ** 2)            
            losses += loss.item()
            n_samples += src.shape[1]
            progress_bar.update(src.shape[1])
            count += 1
            torch.cuda.empty_cache()
            gc.collect()
    
    return losses / count, y_all, tgt_all, n_samples#, memories