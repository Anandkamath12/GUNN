checkpoint = {}

def deepCopy(tbl):
   
   copy = {}
   for k, v in pairs(tbl):
      if type(v) == 'table':
         copy[k] = deepCopy(v)
      else:
         copy[k] = v
      
   
   if torch.typename(tbl):
      torch.setmetatable(copy, torch.typename(tbl))
   
   return copy


def latest(opt):
   if opt.resume == 'none':
      return nil
   

   latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath):
      return nil
   
   print('=> Loading checkpoint ' .. latestPath)
   latest = torch.load(latestPath)
   optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState


def save(epoch, model, optimState, isBestModel, opt)
   
   if torch.type(model) == 'nn.DataParallelTable':
      model = model.get(1)
   

  
   model = deepCopy(model).float().clearState()



   if isBestModel:
      torch.save(paths.concat(opt.save, 'model_best.t7'), model)
   

return checkpoint