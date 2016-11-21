--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local model = 'googlenet'
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    
    ------------ General options --------------------
    cmd:option('-cache', './imagenet/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-manualSeed',           2,      'Manually set RNG seed')
    cmd:option('-GPU',                  1,      'Default preferred GPU')
    ------------- Data options ------------------------
    cmd:option('-dataType',             1,      'select the dataset format, 0 -- jpeg, 1 -- lmdb')
    cmd:option('-nClasses',             1000,   'number of classes in the dataset')
    cmd:option('-backend',              'nn',   'Options: cudnn | nn')


    if model == 'googlenet' then
       ------------- Training options --------------------
       cmd:option('-model',                'googlenet','train or test')
       cmd:option('-phase',                'train','train or test')
       cmd:option('-newRegime',            true,   'use nnlr or not')
       cmd:option('-nEpochs',              250,    'Number of total epochs to run')
       cmd:option('-epochSize',            40000,  'Number of batches per epoch')
       cmd:option('-epochNumber',          1,      'Manual epoch number (useful on restarts)')
       cmd:option('-batchSize',             32,    'mini-batch size (1 = pure stochastic)')
       cmd:option('-prefetchSize',         5,      'prefetch batch size (2 or 3 is recommended)')
       ---------- Optimization options ----------------------
       cmd:option('-optimization',         'sgd',  'optimization method')
       cmd:option('-learningRate',                   0.0,    'learning rate; if set, overrides default LR/WD recipe')
       cmd:option('-momentum',             0.9,    'momentum')
       cmd:option('-weightDecay',          2e-4,   'weight decay')
       ---------- Model options ----------------------------------
       cmd:option('-netType',              'zpGoogle_mkldnn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    elseif model == 'alexnet' then
       cmd:option('-model',                'alexnet','train or test')
       cmd:option('-phase',                'train','train or test')
       cmd:option('-newRegime',            true,   'use nnlr or not')
       cmd:option('-nEpochs',              90,    'Number of total epochs to run')
       cmd:option('-epochSize',            5000,  'Number of batches per epoch')
       cmd:option('-epochNumber',          1,      'Manual epoch number (useful on restarts)')
       cmd:option('-batchSize',            256,    'mini-batch size (1 = pure stochastic)')
       cmd:option('-prefetchSize',         4,      'prefetch batch size (2 or 3 is recommended)')
       ---------- Optimization options ----------------------
       cmd:option('-optimization',         'sgd',  'optimization method')
       cmd:option('-learningRate',                   0.0,    'learning rate; if set, overrides default LR/WD recipe')
       cmd:option('-momentum',             0.9,    'momentum')
       cmd:option('-weightDecay',          5e-4,   'weight decay')
       ---------- Model options ----------------------------------
       cmd:option('-netType',              'alexnet_mkldnn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    end









    cmd:option('-retrain',              'none', 'provide path to model to retrain with')
    cmd:option('-optimState',           'none', 'provide path to an optimState to reload from')

    cmd:text()

    local opt = cmd:parse(arg or {})
    return opt
end

return M
