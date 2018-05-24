import numpy as np


class Data:
    """
    Input data pipeline class
    """
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
        new_ligids = np.empty(self.ligids.shape, dtype=self.ligids.dtype)
        new_smiles = np.empty(self.smiles.shape, dtype=self.smiles.dtype)
        new_scores = np.empty(self.scores.shape, dtype=self.scores.dtype)
        
        perm_ligids = np.random.permutation(self.ligids.shape[0])
        perm_smiles = np.random.permutation(self.smiles.shape[0])
        
        for old_idx, new_idx in enumerate(perm_ligids):
            new_ligids[new_idx] = self.ligids[old_idx]
        for old_idx, new_idx in enumerate(perm_smiles):
            new_smiles[new_idx] = self.smiles[old_idx]
        for old_lig_idx, new_lig_idx in enumerate(perm_ligids):
            for old_smi_idx, new_smi_idx in enumerate(perm_smiles):
                if (old_lig_idx*self.num_smiles+old_smi_idx)%100000 == 0:
                    print('new: {:<10} | old: {:<10}'.format(new_lig_idx*self.num_smiles+new_smi_idx,
                                                             old_lig_idx*self.num_smiles+old_smi_idx), end='\r')
                new_scores[new_lig_idx*self.num_smiles+new_smi_idx] = \
                self.scores[old_lig_idx*self.num_smiles+old_smi_idx]
                
        self.ligids = new_ligids
        self.smiles = new_smiles
        self.scores = new_scores
        
    def reset(self,shuffle=False):
        self.batch_index = 0
        if shuffle:
            self.shuffle()

class Data2:
    """
    The datasets built into Tensorflow allows you to conveniently call the
    next_batch() function to get the next batch of data.
    This is just a reimplementation of that function.
    """
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.num_lig_smi = int(X_data.shape[0])
        self.num_scores = int(Y_data.shape[0])
        self.batch_num = 0
    
    def next_batch(self, batch_size):
        """
        Used for gradient descent when the input data set is too large.
        You can split it up into batches of BATCH_SIZE and iterate through the batches.
        """
        X_batch = self.X_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size]
        Y_batch = self.Y_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size]
        self.batch_num += 1
        return X_batch, Y_batch
    
    def full_batch(self):
        """
        Returns a batch containing all data
        """
        return self.X_data, self.Y_data
    
    def random_batch(self, batch_size):
        """
        Used for stochastic gradient descent.
        Cuts the dataset into batches of BATCH_SIZE and randomly selects one of those batches
        """
        rand_nums = np.random.randint(self.X_data.shape[0], size=(batch_size))
        X_batch = self.X_data[rand_nums]
        Y_batch = self.Y_data[rand_nums]
        return X_batch, Y_batch

    def shuffle(self):
        """
        Shuffle the data between every epoch to have faster convergence
        """
        new_X = np.empty(self.X_data.shape, dtype=self.X_data.dtype)
        new_Y = np.empty(self.Y_data.shape, dtype=self.Y_data.dtype)
        perm = np.random.permutation(self.X_data.shape[0])
        for old_idx, new_idx in enumerate(perm):
            new_X[new_idx] = self.X_data[old_idx]
            new_Y[new_idx]   = self.Y_data[old_idx]
        self.X_data = new_X
        self.Y_data = new_Y
        
    def reset(self, shuffle=False):
        """
        Resets the data. Used after every epoch
        """
        self.batch_num = 0
        if shuffle:
            self.shuffle()
            
########################################################################################

def train_validation_split(ligids, smiles, labels, num_val_lig=3046, num_val_smi=10581, shuffle=True):
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
    if shuffle:
        data.shuffle()
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

def remap_scores(scores, map_from, map_to):
    [print('Remapping {} to {}'.format(_from, _to)) for _from, _to in zip(map_from, map_to)]
    for _from, _to in zip(map_from, map_to):
        scores[scores==_from] = _to
    new_distribution = np.bincount(scores)/scores.shape[0]
    print('New score distribution: {}'.format(new_distribution))
    return scores

def reduce_dimensions(data, lower_thresh, upper_thresh):
    lower_thresh_idx = np.ones(shape=(data.shape[1]))
    upper_thresh_idx = np.ones(shape=(data.shape[1]))
    if lower_thresh is not None:
        lower_thresh_idx =  np.sum(data, axis=0)>lower_thresh
    if upper_thresh is not None:
        upper_thresh_idx =  np.sum(data, axis=0)<upper_thresh
            
    thresh_idx = np.logical_and(lower_thresh_idx, upper_thresh_idx)
    reduced_data = data[:,thresh_idx]
    return reduced_data