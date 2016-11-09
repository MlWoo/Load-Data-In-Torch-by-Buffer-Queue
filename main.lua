local paths = require 'paths'
local tunnel = require 'tunnel'
local lmdb = require 'lmdb' 
require 'nnlr'
-------------------------------------------------------
-- 0
torch.setdefaulttensortype('torch.FloatTensor')

--***************************************************--

--------------------------------------------------------
-- 1. file configuration
-------------------------

local config = paths.dofile('config.lua')

--****************************************************--

--------------------------------------------------------
-- 2. cmd configuration
-----------------------

local opts = paths.dofile('opts.lua')
cmd = opts.parse(arg)
if cmd.model == 'alexnet' then
   config.croppedSize = {3,227,227}
end

--****************************************************--

--------------------------------------------------------
-- 3. create Data configuration
--  (0) phase(0/1)          cmd     train/test 
--  (1) data source(0/1)    cmd     Jpeg/LMDB 
--  (2) data path           config  string                              
--  (3) compress flag       config  true/false
--  (4) cropped image size  config  eg.{3, 224, 224}
--  (5) batchSize           cmd     int:32
--  (6) prefetchSize        config  int:5
--  (6) Normalization       config  eg.{'simple', 118.380948, 61.896913}
--  (7) ImageNetClasses     config  eg.1000
--  (8) ValidationLabels    config  eg.
--  (9) ImageMinSide        config  eg.256                                  --Minimum side length of saved images
--  (10)donkey              config  eg.4
--  (11)access way(0/1)     config  Seq/Random
-----------------------------------------------------
dataConfig = {
    phase           = cmd.phase,
    dataSource      = cmd.dataSource,
    dataPath        = config.path,
    Compressed      = config.Compressed,
    croppedSize     = config.croppedSize,
    batchSize       = cmd.batchSize,
    prefetchSize    = config.prefetchSize,
    Normalization   = config.Normalization,
    ImageNetClasses = config.ImageNetClasses,
    ValLabels       = config.ValidationLabels,
    ImageMinSide    = config.ImageMinSide,
    donkey          = config.donkey,
    accessWay       = config.accessWay,   
    epochs          = cmd.nEpochs, 
    epochSize       = cmd.epochSize
}

--****************************************************--

--------------------------------------------------------
-- 4. create Model configuration
--  (0) phase(0/1)          cmd     train/test 
--  (1) net type            cmd     string
--  (2) manualSeed          cmd     int
--  (3) epochs              cmd     int:250
--  (4) epochSize           cmd     int:40000
--  (5) epochNumber         cmd     int:1
--  (6) batchSize           cmd     int:32
--  (7) learning rate       cmd     float
--  (8) momentum            cmd     float
--  (9) weightDecay         cmd     float
--  (10)GPU/CPU             cmd     1/0
--  (11)nGPU                config  number
--  (12)cache               config  path string
--  (13)backend             config  cudnn | nn
--  (14)retrain             config  path string
--  (15)optimstate          config  path string
-----------------------------------------------------

modelConfig = {
    model           = cmd.model,
    phase           = cmd.phase,
    netType         = cmd.netType,
    optimization    = cmd.optimization,
    mannualSeed     = cmd.mannualSeed,
    epochs          = cmd.nEpochs,
    epochSize       = cmd.epochSize,
    epochNumber     = cmd.epochNumber,
    batchSize       = cmd.batchSize,
    nClasses        = config.nClasses,
    croppedSize     = config.croppedSize,
    learningRate    = cmd.learningRate,
    momentum        = cmd.momentum,
    weightDecay     = cmd.weightDecay,
    GPU_CPU         = cmd.GPU_CPU,
    nGPU            = config.nGPU,
    modelsFolder    = config.modelsFolder,
    cache           = config.cache,
    backend         = config.backend,
    retrain         = config.retrain,
    newRegime 	    = cmd.newRegime,

    optimState      = config.optimState
}
print(modelConfig)
--****************************************************--

--------------------------------------------------------
-- 5. read LMDB dataset
-----------------------------------------------------
paths.dofile('LMDBProvider.lua')
TrainDB = LMDBProvider(dataConfig)

--****************************************************--

--------------------------------------------------------
-- 6. read model
-----------------------------------------------------
paths.dofile('createModel.lua')
TrainModel = netOptim(modelConfig)
print("create model done")
--****************************************************--

--------------------------------------------------------
-- 7. read producer
-----------------------------------------------------
producer = function(vector, printer, DB)
    DB:open()
    printer('produce haha')    
    local epochs = DB.config.epochs
    local batchsize = DB.config.batchSize
    local epochsize = DB.config.epochSize
    local itemNum = batchsize*epochsize

    for i = 1, epochs do
        DB:shuffle(itemNum)
        for j = 1, itemNum do
            Data, Label = DB:cache(j, itemNum)
            vector:pushBack({Data, Label})
--            printer('producer', __threadid, i, j)
        end
    end
    DB:close()
    

end

--****************************************************--

--------------------------------------------------------
-- 8. consumer
--------------------------------------------------------

consumer = function(vector, printer, model)
    print('consume haha')

    local threads = require 'threads'
    local sys = require 'sys'
    mutex = threads.Mutex() 
    torch.setdefaulttensortype('torch.FloatTensor')
    local epochs = model.config.epochs
    local batchSize = model.config.batchSize
    local epochSize = model.config.epochSize
    local croppedSize = model.config.croppedSize
            
    local batchData = torch.Tensor(batchSize, croppedSize[1], croppedSize[2], croppedSize[3]) 
    local batchLabel = torch.Tensor(batchSize)
    local product
    
    local lastTick = nil
    local interval = nil
    for i =1, epochs do
       model:setTrainOptim(i)

        for j =1, epochSize do
            if(j%100 == 0) then
               curTick = sys.clock() 
               if(lastTick ~= nil) then
                  interval = curTick - lastTick
               end
               lastTick = curTick
               printer('train 100 batch time = ', __threadid, j,  interval, ' sec')
            end
                       
            for k =1, batchSize do
                product = vector:popFront()
--                printer('consumer', __threadid,  j, k)
                batchData[k] = product[1]
                batchLabel[k] = product[2]

            end

            mutex:lock()
            totalerr = model:trainBatch(batchData, batchLabel)
            mutex:unlock()
	        printer("epoch=",i,",iteration =",j ,", LR = ", model.optimState.learningRate,", loss = ", totalerr)

        end
--        model:clearState()
--        saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
--        torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

    end

end

--***************************************************--


--------------------------------------------------------
-- 9. initalize environment for threads
init_job = function()
    local lmdb = require 'lmdb'
    local path = require 'paths'
    local nn = require 'nn'
    path.dofile('LMDBProvider.lua')
    path.dofile('createModel.lua')
    require 'nnlr'
    require 'nn' 

end
--***************************************************--

---------------------------------------------------------------
-- 10. create variables shared by producer and consumer threads

vector = tunnel.Vector(dataConfig.prefetchSize * dataConfig.batchSize)
printer = tunnel.Printer()


--create blocks
producer_block = tunnel.Block(1, init_job)
consumer_block = tunnel.Block(1, init_job)
producer_block:add(vector, printer, TrainDB)
consumer_block:add(vector, printer, TrainModel)
-- run threads
producer_block:run(producer)
consumer_block:run(consumer)

--**********************************************************--
print(TrainModel.optimState)








