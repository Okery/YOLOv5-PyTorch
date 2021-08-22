import json
import sys
import time

import torch

from . import distributed
from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, ema):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters
    num_iters = epoch * len(data_loader)

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    o_m = Meter("optimizer")
    e_m = Meter("ema")
    model.train()
    A = time.time()
    for i, data in enumerate(data_loader):
        T = time.time()
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            args.accumulate = max(1, round(r * (64 / args.batch_size - 1) + 1))
            for j, p in enumerate(optimizer.param_groups):
                init = 0.1 if j == 0 else 0 # 0: biases
                p["lr"] = r * (args.lr_epoch - init) + init
                p["momentum"] = r * (args.momentum - 0.9) + 0.9
                   
        images = data.images
        targets = data.targets
        S = time.time()
        if args.amp:
            with torch.cuda.amp.autocast():
                losses = model(images, targets)
        else:
            losses = model(images, targets)
            
        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))
            
        losses = {k: v * args.batch_size for k, v in losses.items()}
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)

        #losses_reduced = distributed.reduce_dict(losses)
        #total_loss_reduced = sum(losses_reduced.values())

        if not torch.isfinite(total_loss):
            print("Loss is {}, stopping training".format(total_loss.item()))
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))
            sys.exit(1)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        if num_iters % args.accumulate == 0:
            S = time.time()
            optimizer.step()
            optimizer.zero_grad()
            o_m.update(time.time() - S)
            
            S = time.time()
            ema.update(model)
            e_m.update(time.time() - S)

        t_m.update(time.time() - T)
        
        num_iters += 1
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, ".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg), end="")
    print("backward: {:.1f}, optimizer: {:.1f}, ema: {:.1f}".format(1000*b_m.avg,1000*o_m.avg,1000*e_m.avg))
    return (m_m.sum + b_m.sum + o_m.sum + e_m.sum) / iters
            

def evaluate(model, data_loader, device, args, generate=True, evaluation=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)
      
    output = ""
    if distributed.get_rank() == 0 and evaluation:
        dataset = data_loader.dataset
        coco_evaluator = CocoEvaluator(dataset.coco)

        results = json.load(open(args.results))

        S = time.time()
        coco_evaluator.accumulate(results)
        print("accumulate: {:.1f}s".format(time.time() - S))
        
        # collect the output of builtin function "print"
        temp = sys.stdout
        sys.stdout = TextArea()

        coco_evaluator.summarize()

        output = sys.stdout
        sys.stdout = temp
        
    if hasattr(args, "distributed") and args.distributed:
        torch.distributed.barrier()
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
    ann_labels = data_loader.dataset.ann_labels
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, data in enumerate(data_loader):
        T = time.time()
        
        images = data.images
        targets = data.targets

        #torch.cuda.synchronize()
        S = time.time()
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs, losses = model(images, targets)
        else:
            outputs, losses = model(images, targets)
        m_m.update(time.time() - S)
        
        if losses and i % 10 == 0:
            print("{}\t".format(i), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))
            
        outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
        predictions = {tgt["image_id"].item(): out for tgt, out in zip(targets, outputs)}
        coco_results.extend(prepare_for_coco(predictions, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    
    S = time.time()
    all_results = distributed.all_gather(coco_results)
    print("all gather: {:.1f}s".format(time.time() - S))
    
    merged_results = []
    for res in all_results:
        merged_results.extend(res)
        
    if distributed.get_rank() == 0:
        json.dump(merged_results, open(args.results, "w"))
        
    return m_m.sum / iters
    
