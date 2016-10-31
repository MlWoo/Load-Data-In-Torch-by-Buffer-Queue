require 'nnlr'

function createModel(nGPU)
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()

   local SpatialConvolution = nn.SpatialConvolutionMKLDNN
   local ReLU = nn.ReLUMKLDNN
   local SpatialMaxPooling = nn.SpatialMaxPoolingMKLDNN
   local SBatchNorm = nn.SpatialBatchNormalizationMKLDNN
   local LRN = nn.LRNMKLDNN

   local conv1 = SpatialConvolution(3,96,11,11,4,4)
   conv1:learningRate('weight', 1)
   conv1:weightDecay('weight',  1)
   conv1:learningRate('bias',   2)
   conv1:weightDecay('bias',    0)
   conv1.weight:normal(0, 0.01)
   conv1.bias:fill(0)
   local conv2 = SpatialConvolution(96,256,5,5,1,1,2,2,2)
   conv2:learningRate('weight', 1)
   conv2:weightDecay('weight',  1)
   conv2:learningRate('bias',   2)
   conv2:weightDecay('bias',    0)
   conv2.weight:normal(0, 0.01)
   conv2.bias:fill(0.1)
   local conv3 = SpatialConvolution(256,384,3,3,1,1,1,1)
   conv3:learningRate('weight', 1)
   conv3:weightDecay('weight',  1)
   conv3:learningRate('bias',   2)
   conv3:weightDecay('bias',    0)
   conv3.weight:normal(0, 0.01)
   conv3.bias:fill(0)
   local conv4 = SpatialConvolution(384,384,3,3,1,1,1,1,2)
   conv4:learningRate('weight', 1)
   conv4:weightDecay('weight',  1)
   conv4:learningRate('bias',   2)
   conv4:weightDecay('bias',    0)
   conv4.weight:normal(0, 0.01)
   conv4.bias:fill(0.1)
   local conv5 = SpatialConvolution(384,256,3,3,1,1,1,1,2)
   conv5:learningRate('weight', 1)
   conv5:weightDecay('weight',  1)
   conv5:learningRate('bias',   2)
   conv5:weightDecay('bias',    0)
   conv5.weight:normal(0, 0.01)
   conv5.bias:fill(0.1)
 
   local fc6 = nn.Linear(256*6*6, 4096)
   fc6:learningRate('weight', 1)
   fc6:weightDecay('weight',  1)
   fc6:learningRate('bias',   2)
   fc6:weightDecay('bias',    0)
   fc6.weight:normal(0, 0.005)
   fc6.bias:fill(0.1)
   local fc7 = nn.Linear(4096, 4096)
   fc7:learningRate('weight', 1)
   fc7:weightDecay('weight',  1)
   fc7:learningRate('bias',   2)
   fc7:weightDecay('bias',    0)
   fc7.weight:normal(0, 0.005)
   fc7.bias:fill(0.1)
   local fc8 = nn.Linear(4096, nClasses)
   fc8:learningRate('weight', 1)
   fc8:weightDecay('weight',  1)
   fc8:learningRate('bias',   2)
   fc8:weightDecay('bias',    0)
   fc8.weight:normal(0, 0.01)
   fc8.bias:fill(0)



   features:add(conv1)       -- 224 -> 55
   features:add(ReLU(true))
   features:add(LRN(5,0.0001,0.75))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(conv2)       --  27 -> 27
   features:add(ReLU(true))
   features:add(LRN(5,0.0001,0.75))
   features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(conv3)      --  13 ->  13
   features:add(ReLU(true))
   features:add(conv4)      --  13 ->  13
   features:add(ReLU(true))
   features:add(conv5)      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6


   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(fc6)
   classifier:add(nn.ReLU())
   classifier:add(nn.Dropout(0.5))
   classifier:add(fc7)
   classifier:add(nn.ReLU())
   classifier:add(nn.Dropout(0.5))
   classifier:add(fc8)
   classifier:add(nn.LogSoftMax())

   features:get(1).gradInput = nil
   --classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 227

   return model
end
