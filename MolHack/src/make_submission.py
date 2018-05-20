import tensorflow as tf
import numpy as np
import argparse
from data_utils import *
import sys

def parse_args():
    # Parse args
    parser = argparse.ArgumentParser()    
    parser.add_argument('--batch_size',
                        type=int, 
                        default=10000,
                        help='name of the model we wish to make submission for')
    parser.add_argument('--model',
                        type=str,
                        help='name of the model we wish to make submission for')
    args = parser.parse_args()
    return args.batch_size, args.model

def load_data():
    test_ligids = np.load('../data/PHARM_TEST_X.npy')
    test_smiles = np.load('../data/PHARM_TEST_SMILES.npy')
    test_data = Data(test_ligids, 
                     test_smiles, 
                     np.empty(shape=(test_ligids.shape[0]* test_smiles.shape[0]), dtype=np.int8))
    return test_data


def main(): 
    batch_size, model = parse_args()

    test_data = load_data()
    num_test_batches = int(test_data.num_scores/batch_size)+1

    # Load model and tensors
    tf.reset_default_graph()
    loader = tf.train.import_meta_graph('../models/{}.meta'.format(model))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loader.restore(sess, '../models/{}'.format(model))
        X, Y, pred_score = tf.get_collection('pred_ops')

        # Make predictions
        predictions = []
        test_data.reset()
        for batch_num in range(num_test_batches):
            ligids_batch, smiles_batch, empty_batch = test_data.next_batch(batch_size)
            lig_smi_batch = np.concatenate((ligids_batch,smiles_batch), axis=1)
            prediction_batch = sess.run(pred_score, feed_dict={X:lig_smi_batch, Y:empty_batch})
            predictions.append(prediction_batch)
            batch_bincount = np.bincount(np.squeeze(prediction_batch.astype(int)), minlength=10)
            print('{:<4}/{:<4}: {}'.format(batch_num, 
                                           num_test_batches,
                                           batch_bincount), end='\r')

        predictions = np.concatenate(predictions, axis=0)

    # Write predictions to file
    print('\nWriting to file')
    with open('../submissions/submission_{}.csv'.format(model), 'w') as out_file:
        out_file.write('ligid_pharmid,score\n')
        for glob_index in range(test_data.num_scores):
            if glob_index%10000 == 0:
                print('{:<10}/{:<10}'.format(glob_index, test_data.num_scores), end='\r')
            out_file.write('{},{}\n'.format(glob_index, int(predictions[glob_index])))

if __name__ == '__main__':
    main()