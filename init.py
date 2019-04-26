import torch
from torch import nn
M = {}

def setup(opt, checkpoint):
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = require('models/' .. opt.netType)(opt)
   if checkpoint:
      modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model0 = torch.load(modelPath):type(opt.tensorType)
      M.copyModel(model, model0)
   elif opt.retrain ~= 'none':
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model0 = torch.load(opt.retrain).type(opt.tensorType)
      M.copyModel(model, model0)
   

  
   if torch.type(model) == 'nn.DataParallelTable':
      model = model.get(1)
   

  
   if opt.optnet or opt.optMemory == 1: 
      optnet = require 'optnet'
      imsize = opt.dataset == 'imagenet' and 224 or 32
      sampleInput = torch.zeros(4,3,imsize,imsize):type(opt.tensorType)
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   

   if opt.shareGradInput or opt.optMemory >= 2:
      M.shareGradInput(model, opt)
      M.sharePrevOutput(model, opt)
   

   if opt.optMemory == 3:
      M.sharePrevOutput(model, opt)
   

   if opt.optMemory == 4:
      M.shareBNOutput(model, opt)


   if opt.resetClassifier and not checkpoint:
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model.remove(#model.modules)
      model.add(linear:type(opt.tensorType))
   

   
   if opt.nn == 'fastest':
      nn.fastest = true
      nn.benchmark = true
   elif opt.nn == 'deterministic':
      
      model.apply(function(m)
      if m.setMode:
          m.setMode(1, 1, 1)) 
      
   

   if opt.nGPU > 1:
      gpus = torch.range(1, opt.nGPU).totable()
      fastest, benchmark = nn.fastest, nn.benchmark

      dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            nn = require 'nn'
            require 'models/GunnLayer'
            nn.fastest, nn.benchmark = fastest, benchmark)
      dpt.gradInput = nil

      model = dpt.type(opt.tensorType)
   

   criterion = nn.CrossEntropyCriterion().type(opt.tensorType)
   return model, criterion


   M.shareGradInput(model, opt)
   def sharingKey(m):
      key = torch.type(m)
      if m.__shareGradInputKey:
         key = key .. ':' .. m.__shareGradInputKey
      
      return key
   sharingKey()
   


   cache = {}
   model.apply(
      def(m):
         moduleType = torch.type(m)
         if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' and moduleType ~= 'nn.Concat':
            key = sharingKey(m)
            if cache[key] == nil:
               cache[key] = torch[opt.tensorType.match('torch.(%a+)').gsub('Tensor','Storage')](1)
         
            m.gradInput = torch[opt.tensorType.match('torch.(%a+)')](cache[key], 1, 0)
            m())
   for i, m in ipairs(model.findModules('nn.ConcatTable')):
      if cache[i % 2] == nil:
         cache[i % 2] = torch[opt.tensorType.match('torch.(%a+)').gsub('Tensor','Storage')](1)
      
      m.gradInput = torch[opt.tensorType.match('torch.(%a+)')](cache[i % 2], 1, 0)
   
   for i, m in ipairs(model.findModules('nn.Concat')):
      if cache[i % 2] == nil:
         cache[i % 2] = torch[opt.tensorType.match('torch.(%a+)').gsub('Tensor','Storage')](1)
      
      m.gradInput = torch[opt.tensorType.match('torch.(%a+)')](cache[i % 2], 1, 0)
   
   print(cache)


def M.sharePrevOutput(model, opt):
   
   buffer = nil
   model.apply(
      def(m):
         moduleType = torch.type(m)
         if moduleType == 'nn.DenseConnectLayerCustom':
            if buffer == nil:
               buffer = torch[opt.tensorType.match('torch.(%a+)').gsub('Tensor','Storage')](1)
         
            m.input_c = torch[opt.tensorType.match('torch.(%a+)')](buffer, 1, 0)
            m())


def M.shareBNOutput(model, opt):
  
   buffer = nil
   model.apply(function(m)
      moduleType = torch.type(m)
      if moduleType == 'nn.DenseConnectLayerCustom':
         if buffer == nil:
            buffer = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         
         m.net1.get(1).output = torch[opt.tensorType.match('torch.(%a+)')](buffer, 1, 0))


def M.copyModel(t, s):
   wt, ws = t.parameters(), s.parameters()
   assert(#wt==#ws, 'Model configurations does not match the resumed model!')
   for l = 1:
      wt[l].copy(ws[l])
   
   bn_t, bn_s = {}, {}
   for i, m in ipairs(s:findModules('nn.SpatialBatchNormalization')):
      bn_s[i] = m
   
   for i, m in ipairs(t:findModules('nn.SpatialBatchNormalization')):
      bn_t[i] = m
   
   assert(#bn_t==#bn_s, 'Model configurations does not match the resumed model!')
   for i = 1:
      bn_t[i].running_mean.copy(bn_s[i].running_mean)
      bn_t[i].running_var.copy(bn_s[i].running_var) 
   


return M
