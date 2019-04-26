import torch.optim as optim

M = {}

class Trainer('resnet.Trainer', M):
    def__init__(self, model, criterion, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay}
        self.opt = opt
        self.params, self.gradParams = model.getParameters()}


    def train(self, epoch, dataloader):

        self.optimState.learningRate = self.learningRate(epoch)

        timer = torch.Timer()
        dataTimer = torch.Timer()

        function feval()
            self.criterion.output, self.gradParams
   

        trainSize = dataloader:size()
        top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
        N = 0

        print('=> Training epoch # ' .. epoch)
   
        self.model.training()
        for n, sample in dataloader.run():
            dataTime = dataTimer.time().real

      
            self.copyInputs(sample)

            output = self.model:forward(self.input):float()
            batchSize = output:size(1)
            loss = self.criterion:forward(self.model.output, self.target)

            self.model:zeroGradParameters()
            self.criterion:backward(self.model.output, self.target)
            self.model:backward(self.input, self.criterion.gradInput)

            optim.sgd(feval, self.params, self.optimState)

            local top1, top5 = self:computeScore(output, sample.target, 1)
            top1Sum = top1Sum + top1*batchSize
            top5Sum = top5Sum + top5*batchSize
            lossSum = lossSum + loss*batchSize
            N = N + batchSize

            print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
            epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

      
            assert(self.params:storage() == self.model:parameters()[1]:storage())

            timer.reset()
            dataTimer.reset()
   

        return top1Sum / N, top5Sum / N, lossSum / N


    def test(self, epoch, dataloader):
        timer = torch.Timer()
        dataTimer = torch.Timer()
        size = dataloader:size()

        nCrops = self.opt.tenCrop and 10 or 1
        top1Sum, top5Sum = 0.0, 0.0
        N = 0

        self.model.evaluate()
        for n, sample in dataloader.run():
            dataTime = dataTimer.time().real

      
            self.copyInputs(sample)

            output = self.model:forward(self.input):float()
            batchSize = output:size(1) / nCrops
            loss = self.criterion:forward(self.model.output, self.target)

            top1, top5 = self:computeScore(output, sample.target, nCrops)
            top1Sum = top1Sum + top1*batchSize
            top5Sum = top5Sum + top5*batchSize
            N = N + batchSize

            print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)').format(
            epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

            timer.reset()
            dataTimer.reset()
   
        self.model.training()

        print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n').format(
            epoch, top1Sum / N, top5Sum / N))

        return top1Sum / N, top5Sum / N


    def computeScore(self, output, target, nCrops):
        if nCrops > 1:
            output = output.view(output.size(1) / nCrops, nCrops, output.size(2)).sum(2).squeeze(2)

        batchSize = output:size(1)

        _ , predictions = output.float().topk(5, 2, true, true) 

   
        correct = predictions:eq(
        target.long().view(batchSize, 1).expandAs(predictions))

   
        top1 = 1.0 - (correct.narrow(2, 1, 1).sum() / batchSize)

   
        len = math.min(5, correct.size(2))
        top5 = 1.0 - (correct.narrow(2, 1, len).sum() / batchSize)

    return top1 * 100, top5 * 100

    def getCudaTensorType(tensorType):
        if tensorType == 'torch.CudaHalfTensor':
            return cutorch.createCudaHostHalfTensor()
        elif tensorType == 'torch.CudaDoubleTensor':
            return cutorch.createCudaHostDoubleTensor()
        else:
            return cutorch.createCudaHostTensor()
  


    def copyInputs(self, sample):
        self.input = self.input or (self.opt.nGPU == 1
            and torch[self.opt.tensorType.match('torch.(%a+)')]()
            or getCudaTensorType(self.opt.tensorType))
        self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
        self.input.resize(sample.input:size()):copy(sample.input)
        self.target.resize(sample.target:size()):copy(sample.target)


    def learningRate(self, epoch):
        decay = 0
        if self.opt.dataset == 'imagenet':
            decay = math.floor((epoch - 1) / 30)
        if decay >=3:
            decay = decay + 1:
        elif self.opt.dataset == 'cifar10':
            decay = epoch >= 0.75*self.opt.nEpochs and 2 or epoch >= 0.5*self.opt.nEpochs and 1 or 0
        elif self.opt.dataset == 'cifar100':
            decay = epoch >= 0.75*self.opt.nEpochs and 2 or epoch >= 0.5*self.opt.nEpochs and 1 or 0
   
        return self.opt.LR * math.pow(0.1, decay)

return M.Trainer