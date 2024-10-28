import torch
from torch.utils.data import DataLoader
import torch
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

def cal_chloride(data):
    """
    Calculate the chloride concentration based on the provided data.

    This function computes the chloride concentration using different formulas 
    depending on the value of the `AANDC` attribute in the `data` object. 
    The formulas account for various components such as Sodium, Potassium, 
    Acetate, AADose, and Phosphate.

    Parameters:
    data (object): An object containing the following attributes:
        - AANDC (str): A code indicating the type of solution.
        - Sodium (float): The concentration of sodium.
        - Potassium (float): The concentration of potassium.
        - Acetate (float): The concentration of acetate.
        - AADose (float): The dose of amino acids.
        - Phosphate (float): The concentration of phosphate.

    Returns:
    float: The calculated chloride concentration.
    """
    if data.AANDC=='0338-0644-06': #travasol
        return data.Sodium+data.Potassium-(data.Acetate-0.88*data.AADose)-(1.47+1.33)/2*data.Phosphate-0.05*data.AADose+0.40*data.AADose
    elif data.AANDC=='0074-2991-05': #unknown
        return data.Sodium+data.Potassium-(data.Acetate-1.50*data.AADose)-(1.47+1.33)/2*data.Phosphate-0.05*data.AADose
    elif data.AANDC=='0074-4166-03': #unknown
        return data.Sodium+data.Potassium-(data.Acetate-2.2*data.AADose)-(1.47+1.33)/2*data.Phosphate-0.05*data.AADose
    else: #trophamine
        return data.Sodium+data.Potassium-(data.Acetate-0.96*data.AADose)-(1.47+1.33)/2*data.Phosphate-0.05*data.AADose

def get_osm(data):
    """
    Calculate the osmolarity of a solution based on various components.

    This function computes the osmolarity (osm) of a solution using the provided data.
    The calculation takes into account multiple factors including amino acid dose, 
    weight, VTBI (Volume To Be Infused), dex dose, fat product type, fat dose, 
    multivitamin dose, phosphate, sodium, potassium, calcium, zinc, magnesium, 
    and levocarnitine.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the following columns:
        - 'AADose': Amino acid dose
        - 'TodaysWeight': Current weight of the patient
        - 'VTBI': Volume to be infused
        - 'DexDose': Dextrose dose
        - 'FatProduct_SMOFlipid_20': Indicator if SMOFlipid 20% is used
        - 'FatProduct_Intralipid_20': Indicator if Intralipid 20% is used
        - 'FatDose': Fat dose
        - 'AdultMVIDose': Adult multivitamin dose
        - 'PedMVIDose': Pediatric multivitamin dose
        - 'Phosphate': Phosphate amount
        - 'Sodium': Sodium amount
        - 'Potassium': Potassium amount
        - 'Calcium': Calcium amount
        - 'Zinc': Zinc amount
        - 'Magnesium': Magnesium amount
        - 'Levocar': Levocarnitine amount
        - 'IVVol': Intravenous volume

    Returns:
    float: The calculated osmolarity of the solution.
    """
    osm = data['AADose'] * data['TodaysWeight'] / 0.1 * 0.866 + \
        data['VTBI'] * data['DexDose']/100 / 0.7 * 3.53 + \
        np.select(
            [
                    data['FatProduct_SMOFlipid_20'] == 1,
                    data['FatProduct_Intralipid_20'] == 1
                ],
                [
                    data['FatDose'] * data['TodaysWeight'] * 0.27 / 0.2,
                    data['FatDose'] * data['TodaysWeight'] * 0.26 / 0.2
                ],
                    default=data['FatDose'] * data['TodaysWeight'] * 0.27 / 0.1
                ) +\
        np.select([data['AdultMVIDose'] > 0],
                [data['AdultMVIDose'] * data['TodaysWeight'] * 4.11],
                default=data['PedMVIDose'] * data['TodaysWeight'] * 0.5
                ) +\
        data['Phosphate'] * data['TodaysWeight'] * 7.2 +\
        (data['Sodium']-data['Phosphate']) * data['TodaysWeight'] * 2 +\
        data['Potassium'] * data['TodaysWeight'] * 2 +\
        data['Calcium'] * data['TodaysWeight']/430.373 * 4 +\
        data['Zinc'] * data['TodaysWeight']/65.38/1000 * 4 +\
        data['Magnesium'] * data['TodaysWeight'] * 4 +\
        data['Levocar']/161.199 * data['TodaysWeight']
    osm = osm/(data['IVVol']/1000)
    return osm

def split_medical_record_nums(group):
    # split the MRN that is more than 14 rows to separate rows
    n = len(group)
    chunks = n // 14
    remainder = n % 14
    new_values = pd.Series(index=group.index, dtype=str)
    for i in range(chunks + 1):  # +1 to handle the remainder
        start_idx = i * 14
        if i < chunks:
            end_idx = start_idx + 14
        else:
            end_idx = start_idx + remainder
        new_values.iloc[start_idx:end_idx] = f"{int(group.name)}{i:02d}"
    return new_values


def create_variable_length_dataloader(X_data, Y_data, meta_data=None, batch_size=16, shuffle=False, PAD_IDX=9999):
    """
    Creates a DataLoader for variable-length sequences with optional metadata.

    Args:
        X_data (list of torch.Tensor): List of input sequences.
        Y_data (list of torch.Tensor): List of target sequences.
        meta_data (list of torch.Tensor, optional): List of metadata sequences. Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to False.
        PAD_IDX (int, optional): Padding index to use for sequences. Defaults to 9999.

    Returns:
        DataLoader: A DataLoader instance that yields batches of padded sequences.
    """
    if meta_data is None:
        data = list(zip(X_data, Y_data))
    else:
        data = list(zip(X_data, Y_data, meta_data))
    def collate_fn(batch):
        if meta_data is None:
            X_batch, Y_batch = zip(*batch)
            X_padded = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=False, padding_value=PAD_IDX)
            Y_padded = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=False, padding_value=PAD_IDX)
            return X_padded, Y_padded
        else:
            X_batch, Y_batch, meta_batch = zip(*batch)
            X_padded = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=False, padding_value=PAD_IDX)
            Y_padded = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=False, padding_value=PAD_IDX)
            meta_padded = torch.nn.utils.rnn.pad_sequence(meta_batch, batch_first=False, padding_value=PAD_IDX)
            return X_padded, Y_padded, meta_padded
    # Determine if shuffling is required
    shuffle = shuffle and (len(data) > batch_size)
    return DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

# for creating padding and attention masks
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
    
def create_mask(src, tgt, PAD_IDX, device):
    """
    Generates masks and padding masks for source and target sequences.

    Args:
        src (torch.Tensor): Source sequence tensor of shape (seq_len, batch_size, ...).
        tgt (torch.Tensor): Target sequence tensor of shape (seq_len, batch_size, ...).
        PAD_IDX (int): Padding index used in the sequences.
        device (torch.device): Device on which the tensors are allocated.

    Returns:
        tuple: A tuple containing:
            - src_mask (torch.Tensor): Source mask tensor of shape (src_seq_len, src_seq_len).
            - tgt_mask (torch.Tensor): Target mask tensor of shape (tgt_seq_len, tgt_seq_len).
            - src_padding_mask (torch.Tensor): Source padding mask tensor of shape (batch_size, src_seq_len).
            - tgt_padding_mask (torch.Tensor): Target padding mask tensor of shape (batch_size, tgt_seq_len).
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = generate_square_subsequent_mask(src_seq_len, device) #torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)[:, :, 0]  #<<-- cos it's 3D need to reduce to 2D
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)[:, :, 0]
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def mse_loss(input, target, ignored_indices, reduction):
    mask = torch.isin(target, ignored_indices)
    out = (input[~mask]-target[~mask])**2
    if reduction == "mean":
        return torch.mean(out)
    elif reduction == "None":
        return out

def pearsonr_corr(input, target, ignored_indices):
    mask = torch.isin(target, ignored_indices)
    input = input[~mask].detach().cpu().numpy()
    target = target[~mask].detach().cpu().numpy()
    return pearsonr(input, target)[0]


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


def vectorized_get_osm(weight, vtbi, AdultMVIDose, 
                       aa_pred, dd_pred, mvi_pred, 
                       po4_pred, na_pred, k_pred, ca_pred,
                       zn_pred, mg_pred, acetate_pred):
    """
    Calculate the osmolarity of a solution based on various input parameters.
    This function computes the osmolarity of a solution using a vectorized approach. 
    It takes into account various components such as amino acids, dextrose, fat, 
    multivitamins, and electrolytes, and adjusts calculations based on specific 
    conditions like the type of fat product and adult multivitamin dose.
    Parameters:
    weight (torch.Tensor): The weight of the patient.
    vtbi (torch.Tensor): Volume to be infused.
    smof (torch.Tensor): Indicator for SMOF lipid emulsion.
    omegaven (torch.Tensor): Indicator for Omegaven lipid emulsion.
    AdultMVIDose (torch.Tensor): Indicator for adult multivitamin dose.
    aa_pred (torch.Tensor): Predicted amino acid concentration.
    dd_pred (torch.Tensor): Predicted dextrose concentration.
    fat_pred (torch.Tensor): Predicted fat concentration.
    mvi_pred (torch.Tensor): Predicted multivitamin concentration.
    po4_pred (torch.Tensor): Predicted phosphate concentration.
    na_pred (torch.Tensor): Predicted sodium concentration.
    k_pred (torch.Tensor): Predicted potassium concentration.
    ca_pred (torch.Tensor): Predicted calcium concentration.
    zn_pred (torch.Tensor): Predicted zinc concentration.
    mg_pred (torch.Tensor): Predicted magnesium concentration.
    levocar_pred (torch.Tensor): Predicted levocarnitine concentration.
    Returns:
    torch.Tensor: The calculated osmolarity of the solution.
    """
    # Initialize osm tensor
    osm = torch.zeros_like(aa_pred)
    
    # Calculate AA factor based on AANDC
    factor = 0.96
    
    # Base calculations
    osm += aa_pred * weight / 0.1 * 0.866
    osm += vtbi * dd_pred/100 / 0.7 * 3.53
    
    # MVI calculations
    adult_cond = AdultMVIDose > 0
    osm[adult_cond] += mvi_pred[adult_cond] * weight[adult_cond] * 4.11
    osm[~adult_cond] += mvi_pred[~adult_cond] * weight[~adult_cond] * 0.5
    
    # Electrolyte calculations
    osm += ca_pred * weight/100 * 0.697
    osm += mg_pred * weight /2 * 246.47 /1000 / 0.5 * 4.06
    osm += zn_pred * weight /1000 /65.38 * 136.286 / 1 * 0.354
    
    # Calculate acetate term
    acetate_term = acetate_pred - factor * aa_pred
    
    # Complex phosphate/potassium/sodium/acetate calculations
    phos_pot_cond1 = (po4_pred > 0) & (k_pred > po4_pred)
    phos_pot_cond2 = (po4_pred > 0) & (k_pred < po4_pred)
    
    # Condition 1: Phosphate > 0 and Potassium > Phosphate
    cond1_calc = (
        po4_pred * weight /3 * 7.4 +
        (k_pred - po4_pred) * weight /2 * 4 +
        acetate_term * weight /2 * 4 +
        (na_pred - acetate_term) * weight /4 * 8
    )
    
    # Condition 2: Phosphate > 0 and Potassium < Phosphate
    cond2_calc = (
        k_pred * weight /4.4 * 7.4 +
        (po4_pred - k_pred) * weight /3 * 7 +
        acetate_term * weight /2 * 4 +
        (na_pred - acetate_term - (po4_pred - k_pred)) * weight /4 * 8
    )
    
    # Default calculation
    default_calc = (
        acetate_term * weight /2 * 4 +
        k_pred * weight /2 * 4 +
        (na_pred - acetate_term) * weight /4 * 8
    )
    
    # Combine conditions
    complex_term = torch.where(phos_pot_cond1, cond1_calc,
                    torch.where(phos_pot_cond2, cond2_calc,
                    default_calc))
    
    osm += complex_term
    
    # Final normalization
    osm = osm/(vtbi/1000)
    
    return osm


def vectorized_cal_chloride(na_pred, k_pred, acetate_pred, aa_pred, po4_pred, aandc):
    """
    Calculate the chloride concentration based on given predictions and conditions.

    This function computes the chloride concentration using a vectorized approach
    based on the provided sodium (na_pred), potassium (k_pred), acetate (acetate_pred),
    amino acids (aa_pred), phosphate (po4_pred) predictions, and a condition array (aandc).

    Parameters:
    na_pred (torch.Tensor): Predicted sodium values.
    k_pred (torch.Tensor): Predicted potassium values.
    acetate_pred (torch.Tensor): Predicted acetate values.
    aa_pred (torch.Tensor): Predicted amino acids values.
    po4_pred (torch.Tensor): Predicted phosphate values.
    aandc (torch.Tensor): Condition array indicating the type of solution.

    Returns:
    torch.Tensor: Calculated chloride concentrations.
    """
    # Create a tensor of zeros with the same shape as other inputs
    chloride = torch.zeros_like(na_pred)
    # Define the conditions
    cond_travasol = (aandc == 0)
    cond_unknown1 = (aandc == 1)
    cond_unknown2 = (aandc == 2)
    # Calculate the common part
    common_part = na_pred + k_pred - (1.47 + 1.33) / 2 * po4_pred - 0.05 * aa_pred
    # Apply the conditions
    chloride[cond_travasol] = common_part[cond_travasol] - (acetate_pred[cond_travasol] - 0.88 * aa_pred[cond_travasol]) + 0.40 * aa_pred[cond_travasol]
    chloride[cond_unknown1] = common_part[cond_unknown1] - (acetate_pred[cond_unknown1] - 1.50 * aa_pred[cond_unknown1])
    chloride[cond_unknown2] = common_part[cond_unknown2] - (acetate_pred[cond_unknown2] - 2.2 * aa_pred[cond_unknown2])
    # For all other cases (trophamine)
    mask_other = ~(cond_travasol | cond_unknown1 | cond_unknown2)
    chloride[mask_other] = common_part[mask_other] - (acetate_pred[mask_other] - 0.96 * aa_pred[mask_other])
    return chloride


def get_variable_indices(vars_array):
    """
    Given an array of variables, this function returns a dictionary mapping each variable 
    to its first occurrence index in the array.

    Args:
        vars_array (array-like): An array-like structure containing the variables.

    Returns:
        dict: A dictionary where keys are the variables from the input array and values 
              are their corresponding first occurrence indices in the array.
    """
    vars_array = np.array(vars_array)
    result = {}
    all_indices = np.where(np.isin(vars_array, vars_array))[0]
    for name in vars_array:
        result[name] = all_indices[vars_array[all_indices] == name][0]
    return result

def vectorized_r(gt, preds):
    """
    Calculate the Pearson correlation coefficient between ground truth and predictions in a vectorized manner.

    Parameters:
    gt (numpy.ndarray): Ground truth values, expected to be a 2D array where each row represents a different sample.
    preds (numpy.ndarray): Predicted values, expected to be a 2D array where each row represents a different sample.

    Returns:
    numpy.ndarray: A 1D array containing the Pearson correlation coefficients for each sample.
    """
    gt_mean = gt.mean(axis=1, keepdims=True)
    preds_mean = preds.mean(axis=1, keepdims=True)
    gt_std = gt.std(axis=1, keepdims=True)
    preds_std = preds.std(axis=1, keepdims=True)
    return ((gt - gt_mean) * (preds - preds_mean)).mean(axis=1) / (gt_std * preds_std).squeeze()

def preprocess_data(data):
    data = data.rename(columns={'FatProduct_SMOFlipid 20%': 'FatProduct_SMOFlipid_20', 
                                'FatProduct_Intralipid 20%': 'FatProduct_Intralipid_20', 
                                'FatProduct_Omegaven 10%': 'FatProduct_Omegaven_10'})
    data['MVIDose'] = data.PedMVIDose + data.AdultMVIDose
    data['Famotidine'] = data['Famotidine'] + data['Ranitidine']/3.75
    data['Chloride'] = data.apply(cal_chloride, axis=1)
    data['AANDC_code'] = data.AANDC.apply(lambda x: 0 if x=='0338-0644-06' else 1 if x=='0074-2991-05' else 2 if x=='0074-4166-03' else 3)
    data['IVVol'] = data.VTBI + data.apply(lambda x: x['FatDose'] * x['TodaysWeight']/0.1 if x['FatProduct_Omegaven_10']==1 else \
                                                     x['FatDose'] * x['TodaysWeight']/0.2, axis=1) # data['FluidDose'] = data['IVVol'] + data['EnteralTotal']
    data['Osm'] = data.apply(get_osm, axis=1)
    data['Ca_g'] = data['Calcium'] * data['TodaysWeight']
    data['soln_factor'] = ((data.Calcium/430.373*data.TodaysWeight*2/data.VTBI*1000)**0.863 * (data.Phosphate*data.TodaysWeight/data.VTBI*1000)**1.19)/(data.AADose*data.TodaysWeight/data.VTBI*100)
    data.MedicalRecordNum = data.MedicalRecordNum.astype(float)
    data = data.sort_values(['MedicalRecordNum', 'day_since_birth'])
    data['sMRN'] = data.copy().groupby('MedicalRecordNum').apply(split_medical_record_nums).reset_index(level=0, drop=True)
    data = data.set_index('sMRN').reset_index()
    return data