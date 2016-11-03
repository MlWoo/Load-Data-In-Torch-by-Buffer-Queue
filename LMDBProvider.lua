require 'lmdb'
local LMDBProvider = torch.class('LMDBProvider')

function ExtractFromLMDBTrain(data, key, config)
    require 'image'
    local reSample = function(sampledImg)
        local sizeImg = sampledImg:size()
        local szx = torch.random(math.ceil(sizeImg[3]/4))
        local szy = torch.random(math.ceil(sizeImg[2]/4))
        local startx = torch.random(szx)
        local starty = torch.random(szy)
        return image.scale(sampledImg:narrow(2,starty,sizeImg[2]-szy):narrow(3,startx,sizeImg[3]-szx),sizeImg[3],sizeImg[2])
    end
    local rotate = function(angleRange)
        local applyRot = function(Data)
            local angle = torch.randn(1)[1]*angleRange
            local rot = image.rotate(Data,math.rad(angle),'bilinear')
            return rot
        end
        return applyRot
    end
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]

    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end
    if math.min(img:size(2), img:size(3)) ~= config.ImageMinSide then
        img = image.scale(img, '^' .. config.ImageMinSide)
    end
    if config.Augment == 3 then
        img = rotate(0.1)(img)
        img = reSample(img)
    elseif config.Augment == 2 then
        img = reSample(img)
    end
    local startX = math.random(img:size(3)-config.croppedSize[3]+1)
    local startY = math.random(img:size(2)-config.croppedSize[2]+1)
    img = img:narrow(3,startX,config.croppedSize[3]):narrow(2,startY,config.croppedSize[2])
    local hflip = torch.random(2)==1
    if hflip then
        img = image.hflip(img)
    end

--    print("ExtractFromLMDBTrain end")
    return img, class
end


function ExtractFromLMDBTest(data, key, config)
    require 'image'
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end

    if (math.min(img:size(2), img:size(3)) ~= config.ImageMinSide) then
        img = image.scale(img, '^' .. config.ImageMinSide)
    end

    local startX = math.ceil((img:size(3)-config.InputSize[3]+1)/2)
    local startY = math.ceil((img:size(2)-config.InputSize[2]+1)/2)
    img = img:narrow(3,startX,config.InputSize[3]):narrow(2,startY,config.InputSize[2])
    return img, class
end


local function Key(num)
    return string.format('%07d',num)
end


function LMDBProvider:__init(config)
    assert(config.phase == 'train' or config.phase == 'test')

    dataWholePath = config.dataPath .. config.phase
    self.Source = lmdb.env({Path = dataWholePath, RDONLY = true})
    
    if (config.phase == 'train') then
        self.ExtractFunction = ExtractFromLMDBTrain
    else
        self.ExtractFunction = ExtractFromLMDBTest
    end

    self.config = config
end


function LMDBProvider:open()
    self.Source:open()
    self.txn = self.Source:txn(true)
    self.cursor = self.txn:cursor()
    start_pos = 1
    self.cursor:set(Key(start_pos))
end

function LMDBProvider:close()
    self.cursor:close()
    self.txn:abort()
    self.Source:close()

end

function LMDBProvider:cacheSeq(pos, itemNum)
    local key, data = self.cursor:get()
    local Data, Labels = self.ExtractFunction(data, key, self.config)
    if (pos < itemNum) then
       self.cursor:next()
    elseif(pos == itemNum) then
       self.cursor:first()
    end
    return Data, Labels
end

function Keys(tensor)
--    print('Keys')
    local tbl = {}
    for i=1,tensor:size(1) do
        tbl[i] = Key(tensor[i])
    end
--    print('/Keys')
    return tbl
end


function LMDBProvider:shuffle(num)
    print(self.config.accessWay)
--    print('shuffle')
    if(self.config.accessWay == 'rand') then --shuffle batches from LMDB
--    print('shuffle')
        local SizeData = num
    
        local dataIndices = torch.range(1, SizeData, 1):long()
    
    --    print(dataIndices:size(1))
        if (self.config.phase == 'train') then -- and (config.accessWay == 'rand') then --shuffle batches from LMDB
            dataIndices = dataIndices:index(1, torch.randperm(dataIndices:size(1)):long())
        end
        self.shuffleIndices = Keys(dataIndices)
    else
        self.shuffleIndices = nil
    end
--    print('/shuffle')
    
end

function LMDBProvider:cache(pos, itemNum)
--    print('cache')
    if (self.config.accessWay == 'rand') then --shuffle batches from LMDB
        return self:cacheRand(pos)
    else
        return self:cacheSeq(pos, itemNum)
    end
--    print('/cache')
end

function LMDBProvider:cacheRand(key_pos)
--    print('cacheRand')
    local value = self.shuffleIndices[key_pos]
--    print(value)
    local item = self.txn:get(value)
    local Data, Labels = self.ExtractFunction(item, key, self.config)
--    print('/cacheRand')
    return Data, Labels
end
