
----------------------------------------------------------------
--  1. Data configuration
-------------------------
local ImageNetClasses = torch.load('./ImageNetClasses')
local ValidationLabels  = torch.load('./ValidationLabels')


for i=1001,#ImageNetClasses.ClassName do
    ImageNetClasses.ClassName[i] = nil
end

--dataPath = '/data/user/xiaohui/LMDB2/debug-byte/'
dataPath = '/data/user/mlwu/LMDB/'

config = 
{
    path = dataPath,
    nClasses = 1000,
    croppedSize = {3, 224, 224},
    prefetchSize = 100,
    ImageMinSide = 256, --Minimum side length of saved images
    ValidationLabels = ValidationLabels,
    ImageNetClasses = ImageNetClasses,
    Normalization = {'simple', 118.380948, 61.896913}, --Default normalization -global mean, std
    Compressed = true,
    donkey = 4,
    accessWay = 'seq'       -- 0-seq / 1-random
}

--************************************************************--


----------------------------------------------------------------
--  2. Model  configuration
----------------------------
config['modelsFolder']      = './models/'
config['nGPU']          = 1
config['cache']         = '/home/... .../checkpoint'
config['backend']       = 'nn'
config['retrain']       = 'none'
config['optimState']    = 'none'


--************************************************************--


return config








