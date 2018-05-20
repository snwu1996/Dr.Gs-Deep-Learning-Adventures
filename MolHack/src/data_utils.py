import numpy as np


class Data:
    def __init__(self, ligids, smiles, scores):
        self.ligids = ligids
        self.smiles = smiles
        self.scores = scores
        self.num_ligids = int(ligids.shape[0])
        self.num_smiles = int(smiles.shape[0])
        self.num_scores = int(scores.shape[0])
        self.batch_index = 0 # Current batch index
    
        assert(self.num_ligids*self.num_smiles == self.num_scores),\
        'number of ligids times number of smiles must equal number of scores'
    
    def next_batch(self, batch_size):                
        assert(self.batch_index < self.num_scores), \
        'batch index out of bound, try doing Data.reset() after stepping through the entire dataset'
        
        if self.batch_index+batch_size > self.num_scores:
            batch_size = self.num_scores-self.batch_index
        
        lig_idx_lower = int(self.batch_index/self.num_smiles)
        lig_idx_upper = int((self.batch_index+batch_size-1)/self.num_smiles)
        smi_idx_lower = self.batch_index-lig_idx_lower*self.num_smiles
        smi_idx_upper = smi_idx_lower+batch_size
        smi_idx_upper -= int((smi_idx_upper-1)/self.num_smiles)*self.num_smiles
        
        if lig_idx_upper-lig_idx_lower == 0:
            ligids_batch = self.ligids[lig_idx_lower,:]
            ligids_batch = np.tile(ligids_batch, (batch_size,1))
            smiles_batch = self.smiles[smi_idx_lower:smi_idx_upper,:]
            
        if lig_idx_upper-lig_idx_lower == 1:
            ligids_batch1 = self.ligids[lig_idx_lower,:]
            ligids_batch1 = np.tile(ligids_batch1, (self.num_smiles-smi_idx_lower,1))
            ligids_batch2 = self.ligids[lig_idx_upper,:]
            ligids_batch2 = np.tile(ligids_batch2, (smi_idx_upper,1))
            ligids_batch = np.concatenate((ligids_batch1,ligids_batch2), axis=0)

            smiles_batch1 = self.smiles[smi_idx_lower:,:]
            smiles_batch2 = self.smiles[:smi_idx_upper,:]
            smiles_batch = np.concatenate((smiles_batch1,smiles_batch2), axis=0)
           
        if lig_idx_upper-lig_idx_lower >= 2:
            raise Exception('batch size too large')
           
        scores_batch = self.scores[self.batch_index:self.batch_index+batch_size]
        self.batch_index += batch_size
        return ligids_batch, smiles_batch, scores_batch

    def full_batch(self):
        raise NotImplementedError('full_batch not implemented')
    
    def random_batch(self, batch_size):
        raise NotImplementedError('random_batch not implemented')
    
    def shuffle(self):
        raise NotImplementedError('shuffle not implemented')
    
    def reset(self,shuffle=False):
        self.batch_index = 0
        if shuffle:
            self.shuffle()
            
########################################################################################

def train_validation_split(ligids, smiles, labels, num_val_lig=3046, num_val_smi=10581):
    """
    Example usage:
        train_data, validation_data = train_validation_split(train_valid_ligids,
                                                             train_valid_smiles,
                                                             train_valid_scores,
                                                             num_val_lig=3046, 
                                                             num_val_smi=10581)
    """
    # Train valiatation split - X data
    num_train_lig = ligids.shape[0]-num_val_lig
    num_train_smi = smiles.shape[0]-num_val_smi

    print('num validation ligids: {}'.format(num_val_lig))
    print('num train ligids: {}'.format(num_train_lig))
    print('num validation smiles: {}'.format(num_val_smi))
    print('num train smiles: {}'.format(num_train_smi))

    train_ligids = ligids[:num_train_lig,:]
    train_smiles = smiles[:num_train_smi,:]
    validation_ligids = ligids[num_train_lig:,:]
    validation_smiles = smiles[num_train_smi:,:]

    # Train validation split - Y data
    train_labels = []
    validation_labels = []
    data = Data(ligids, smiles, labels)
    for lig_num in range(num_train_lig): # Train labels
        _, _, train_labels_batch = data.next_batch(num_train_smi)
        _, _, _ = data.next_batch(num_val_smi)
        train_labels.append(train_labels_batch)
    for lig_num in range(num_val_lig): # Validation labels
        _, _, _ = data.next_batch(num_train_smi)
        _, _, validation_labels_batch = data.next_batch(num_val_smi)
        validation_labels.append(validation_labels_batch)
    train_labels = np.concatenate(train_labels, axis=0)
    validation_labels = np.concatenate(validation_labels, axis=0)
    print('num validation labels: {}'.format(validation_labels.shape[0]))
    print('num train labels: {}'.format(train_labels.shape[0]))

    # Return train and validation datasets
    train_data = Data(train_ligids, train_smiles, train_labels)
    validation_data = Data(validation_ligids, validation_smiles, validation_labels)
    return train_data, validation_data