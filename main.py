
import torch
import torch.optim as optim
from torch import nn
import paths
DataLoader = require 'dataloader'
models = require 'models/init'
Trainer = require 'train'
opts = require 'opts'
checkpoints = require 'checkpoints'


torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)


checkpoint, optimState = checkpoints.latest(opt)


model, criterion = models.setup(opt, checkpoint)


trainLoader, valLoader = DataLoader.create(opt)


trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly:
   top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return


startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
bestTop1 = math.huge
bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs:
    trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
    testTop1, testTop5 = trainer:test(epoch, valLoader)
    bestModel = false
    if testTop1 < bestTop1:
        bestModel = true
        bestTop1 = testTop1
        bestTop5 = testTop5
        print(' * Best model ', testTop1, testTop5)
   
    checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)


print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))