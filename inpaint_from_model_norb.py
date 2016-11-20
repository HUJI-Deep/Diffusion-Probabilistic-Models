#!/usr/bin/env python

"""
Use trained model to apply inpainting to NORB-Small dataset with missing data.

Dataset details:
- Dataset should be saved in the Caffe image data format- image files and index file with path
of each image.
- Missing data mask information must also be supplied, also in Caffe image data format. Mask
images should be boolean, true on missing pixels and false for observed pixels.

"""
import argparse
import numpy as np
import os
import sys
from os.path import join
import warnings
import progressbar
import PIL.Image
from scipy.misc import imsave
from scipy.misc import toimage
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.algorithms import (RMSProp, GradientDescent, CompositeRule,
    RemoveNotFinite)
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
import blocks.model
from blocks.roles import INPUT, PARAMETER

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift

import extensions
import model
import util
from sampler import inpaint_masked_samples

from theano.misc import pkl_utils

def save_single_image(x, save_dir, save_name ): 
    x = x.transpose((1,2,0))
    toimage(x, cmin=0.0, cmax=1.0).save(join(save_dir,save_name))


def load_image(path, scale=255.0):
    return np.float32(PIL.Image.open(path)) / scale

def load_images(images_dir, image_size, index_file, flatten=True):
    if not os.path.exists(images_dir):
        raise IOError('Error- %s doesn\'t exist!' % images_dir)
    raw_im_data = np.loadtxt(os.path.join(images_dir,index_file),delimiter=' ',dtype=str)
    total_images = raw_im_data.shape[0]
    
    if flatten:
        ims = np.zeros((total_images,np.product(image_size)))
    else:
        ims = np.zeros((total_images, 3, 32, 32 ))

    for idx in np.arange(total_images):
        print ('loading image %d of %d  \r' % (idx+1,total_images)),
        sys.stdout.flush() 
        if flatten:
            ims[idx,:] = load_image(os.path.join(images_dir,raw_im_data[idx][0])).reshape(np.product(image_size))
        else:
            im = load_image(os.path.join(images_dir,raw_im_data[idx][0]))
            im = im.transpose((2,0,1))
            ims[idx,:,:,:] = im
    print "\n"
    return ims

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--resume_file', default=None, type=str,
                        help='Name of saved model to continue training', required=True)
    parser.add_argument('--missing_dataset_path', default=None, type=str,
                        help='Path to dir containing index.txt and index_mask.txt')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    db_dir = args.missing_dataset_path 

    new_db_path = db_dir + '_dpm_ip'
    if not os.path.exists(new_db_path):
            os.mkdir(new_db_path)

    t_length = 1000 # Trajectory length for inpainting procedure

    batch_size = args.batch_size
    norb_size = (32,32,3)

    load = pkl_utils.load
    print "Resuming training from " + args.resume_file
    with open(args.resume_file, "rb") as f:
        main_loop = load(f)

        # To get the model, assuming it was trained with train_norb.py
        ll_ext = main_loop.extensions[2]
        dpm = ll_ext.model
        X_noisy = T.tensor4('X noisy samp', dtype=theano.config.floatX)
        t = T.matrix('t samp', dtype=theano.config.floatX)
        get_mu_sigma = theano.function([X_noisy, t], dpm.get_mu_sigma(X_noisy, t),
            allow_input_downcast=True)

        input_h = dpm.spatial_width
        input_w = dpm.spatial_width      
        raw_im_data = np.loadtxt(os.path.join(db_dir,'index.txt'),delimiter=' ',dtype=str)
        
        print "Loading images to inpaint..."
        X = load_images(db_dir,(input_h,input_w),index_file='index.txt',flatten=False)
        # Caluclated on whole dataset according to 
        # scl = 1./np.sqrt(np.mean((X-np.mean(X))**2))
        # shft = -np.mean(X*scl)
        scl = 2.613
        shft = -1.31
        
        print "Loading masks..."
        X_mask = load_images(db_dir,(input_h,input_w),index_file='index_mask.txt',flatten=False).astype(bool)
        
        # Same method as in original code except we're calculating on whole dataset and not just per minibatch
        X_scale_shift = X * scl + shft
        N = raw_im_data.shape[0]
        n_batches = N / batch_size

        

        dpm.trajectory_length = t_length
        pbar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('\rProcessed %(value)d of %(max)d Batches '), progressbar.Bar()], maxval=n_batches, term_width=50).start()
        with open(join(new_db_path,'index.txt'),'wb') as db_file:
            for b in np.arange(n_batches):
                X_batch = X_scale_shift[b*batch_size:(b+1)*batch_size,:,:,:]
                X_batch_mask = np.logical_not(X_mask[b*batch_size:(b+1)*batch_size,:,:,:].astype(bool))
                X0 = inpaint_masked_samples(dpm, get_mu_sigma, X_batch, X_batch_mask.ravel())
                X0 = (X0 - shft) / scl

                for idx in np.arange(batch_size):
                    abs_idx = (b*batch_size)+idx
                    save_name = raw_im_data[abs_idx][0].replace('corrupted','ip')
                    save_single_image(X0[idx,:,:,:], new_db_path,save_name)
                    db_file.write('%s %s\n' % ( save_name, raw_im_data[abs_idx][1]))

                pbar.update(b)

        pbar.finish()





        
        