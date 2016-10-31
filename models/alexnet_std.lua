function createModel(nGPU)
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()

   local SpatialConvolution = nn.SpatialConvolutionMKLDNN
   local ReLU = nn.ReLUMKLDNN
   local SpatialMaxPooling = nn.SpatialMaxPoolingMKLDNN
   local SBatchNorm = nn.SpatialBatchNormalizationMKLDNN
   local LRN = nn.LRNMKLDNN

   features:add(SpatialConvolution(3,96,11,11,4,4,0,0):reset(0.01))       -- 224 -> 55
   features:add(ReLU(true))
   features:add(LRN(5,0.0001,0.75))
   --features:add(SBatchNorm(96))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(SpatialConvolution(96,256,5,5,1,1,2,2,2):reset(0.01))       --  27 -> 27
   features:add(ReLU(true))
   features:add(LRN(5,0.0001,0.75))
   --features:add(SBatchNorm(256))
   features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(SpatialConvolution(256,384,3,3,1,1,1,1):reset(0.01))      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialConvolution(384,384,3,3,1,1,1,1):reset(0.01))      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialConvolution(384,256,3,3,1,1,1,1):reset(0.01))      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6


   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Linear(256*6*6, 4096):reset(0.005))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096):reset(0.005))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, nClasses):reset(0.01))
   classifier:add(nn.LogSoftMax())

   features:get(1).gradInput = nil
   --classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 227

   return model
end
