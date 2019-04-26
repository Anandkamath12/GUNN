from torchvision.datasets import datasets 
import Threds
Threads.serialization('threads.sharedserialize')

M = {}
DataLoader = torch.class('resnet.DataLoader', M)
class DataLoader:
   def create(opt):
      loaders = {}
      for i, split in ipairs{'train', 'val'}:
      dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   

      return table.unpack(loaders)

   def __init__(self, dataset, opt, split):
      manualSeed = opt.manualSeed
      def init():
         require('datasets/' .. opt.dataset)
   
      def main(idx):
         if manualSeed ~= 0:
            torch.manualSeed(manualSeed + idx)
         torch.setnumthreads(1)
         _G.dataset = dataset
         _G.preprocess = dataset.preprocess()
         return dataset.size()
   

      threads, sizes = Threads(opt.nThreads, init, main)
      self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
      self.threads = threads
      self.__size = sizes[1][1]
      self.batchSize = math.floor(opt.batchSize / self.nCrops)
      def getCPUType(tensorType):
         if tensorType == 'torch.CudaHalfTensor':
            return 'HalfTensor'
         elif tensorType == 'torch.CudaDoubleTensor':
            return 'DoubleTensor'
         else
            return 'FloatTensor'
      self.cpuType = getCPUType(opt.tensorType)


   def size():
      return math.ceil(self.__size / self.batchSize)


   def run():
      threads = self.threads
      size, batchSize = self.__size, self.batchSize
      perm = torch.randperm(size)

      idx, sample = 1, nil
      def enqueue():
         while idx = size and threads.acceptsjob():
            indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            threads.addjob(
            def(indices, nCrops, cpuType):
               sz = indices.size(1)
               batch, imageSize
               target = torch.IntTensor(sz)
               for i, idx in ipairs(indices.totable()):
                  sample = _G.dataset.get(idx)
                  input = _G.preprocess(sample.input)
                  if not batch:
                     imageSize = input.size().totable()
                     if nCrops > 1:
                        table.remove(imageSize, 1) 
                     batch = torch[cpuType](sz, nCrops, table.unpack(imageSize))
                  
                  batch[i].copy(input)
                  target[i] = sample.target
               
               collectgarbage()
               return 
               {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target = target,
               }
            def(_sample_)
               sample = _sample_,
            
            indices,
            self.nCrops,
            self.cpuType
         )
         idx = idx + batchSize
      

   n = 0
   def loop():
      def enqueue():
         if not threads.hasjob():
            return nil
      
         threads.dojob()
         if threads.haserror():
            threads.synchronize()
      
      enqueue()
      n = n + 1
      return n, sample
   

   return loop


return M.DataLoader