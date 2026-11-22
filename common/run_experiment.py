"""Shared scene-held-out training and evaluation for three Grasp-Anything++ pilots."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset


SIZE = 160
LENGTH = 20
MAX_TARGETS = 32
WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.I)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=121)
    p.add_argument('--lr', type=float, default=4e-4)
    return p.parse_args()


def safe_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_method(path):
    spec = importlib.util.spec_from_file_location('grasp_method', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def vocab_for(rows):
    words = Counter(w.lower() for row in rows for w in WORD.findall(row['prompt']))
    return {w:i for i,(w,_) in enumerate(sorted(words.items(), key=lambda v:(-v[1],v[0]))[:8189], 3)}


def tokenize(prompt, vocabulary):
    tokens = [vocabulary.get(w.lower(),1) for w in WORD.findall(prompt)][:LENGTH] or [1]
    return torch.tensor(tokens+[0]*(LENGTH-len(tokens)), dtype=torch.long)


def normalized(rows, width, height):
    scale = max(width,height)
    ox, oy = (scale-width)*.5, (scale-height)*.5
    boxes = np.asarray(rows,dtype=np.float32).reshape(-1,5).copy()
    boxes[:,0] = (boxes[:,0]+ox)/scale
    boxes[:,1] = (boxes[:,1]+oy)/scale
    boxes[:,2:4] /= scale
    if not len(boxes) or not np.isfinite(boxes).all() or not np.all(boxes[:,2:4]>0):
        raise ValueError('Invalid grasp annotation')
    return boxes


class GraspDataset(Dataset):
    def __init__(self, rows, images, vocab, train=False):
        self.rows=rows
        self.images=images
        self.vocab=vocab
        self.train=train
        self.tokens=[tokenize(r['prompt'],vocab) for r in rows]
        groups=defaultdict(list)
        for i,row in enumerate(rows):
            groups[row['scene_id']].append(i)
        self.alt=[]
        self.same_scene=[]
        for i,row in enumerate(rows):
            same=next((j for j in groups[row['scene_id']]
                       if rows[j]['object_id']!=row['object_id'] and rows[j]['prompt']!=row['prompt']), None)
            self.same_scene.append(same)
            other=same
            if other is None:
                for advance in range(1,min(len(rows),32)):
                    j=(i+advance)%len(rows)
                    if rows[j]['scene_id']!=row['scene_id'] and rows[j]['prompt']!=row['prompt']:
                        other=j
                        break
            self.alt.append(i if other is None else other)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(self.images/row['image']) as f:
            im=f.convert('RGB')
        w,h=im.size
        boxes=normalized(row['grasps'],w,h)[:MAX_TARGETS]
        if im.width != im.height:
            padded=Image.new('RGB',(max(w,h),)*2)
            padded.paste(im,((max(w,h)-w)//2,(max(w,h)-h)//2))
            im=padded
        im=im.resize((SIZE,SIZE),Image.Resampling.BILINEAR)
        if self.train and not re.search(r'\b(?:left|right|clockwise|counterclockwise)\b',row['prompt'],re.I) and random.random()<0.5:
            im=ImageOps.mirror(im)
            boxes[:,0]=1.-boxes[:,0]
            boxes[:,4]=-boxes[:,4]
        image=torch.from_numpy(np.array(im,dtype=np.uint8).transpose(2,0,1).copy()).float()/255.
        targets=torch.zeros(MAX_TARGETS,5)
        targets[:len(boxes)]=torch.from_numpy(boxes)
        valid=torch.arange(MAX_TARGETS)<len(boxes)
        return image,self.tokens[i],targets,valid,i


def cv_rect(box):
    x,y,w,h,theta=map(float,box)
    return ((x*SIZE,y*SIZE),(max(w*SIZE,1e-4),max(h*SIZE,1e-4)),math.degrees(theta))


def correct(box, truth):
    if not np.isfinite(box).all() or min(box[2:4])<=0:
        return False
    d=np.abs(((truth[:,4]-box[4]+math.pi/2)%math.pi)-math.pi/2)
    first=cv_rect(box)
    area=float(box[2]*box[3])*SIZE**2
    for gt in truth[d<math.pi/6]:
        second=cv_rect(gt)
        _,intersection=cv2.rotatedRectangleIntersection(first,second)
        if intersection is None:
            continue
        overlap=cv2.contourArea(intersection)
        union=area+float(gt[2]*gt[3])*SIZE**2-overlap
        if union>0 and overlap/union>0.25:
            return True
    return False


def evaluate(method,net,loader,device,counterfactual=False):
    net.eval()
    count=top1=top5=shuffled=pair_count=pair_correct=0
    losses=[]
    predictions=[]
    seconds=0.
    data=loader.dataset
    with torch.inference_mode():
        for x,t,y,m,ix in loader:
            x,t,y,m=(v.to(device,non_blocking=True) for v in (x,t,y,m))
            torch.cuda.synchronize(device)
            start=time.perf_counter()
            output=net(x,t)
            boxes=method.decode(output,topk=5)
            torch.cuda.synchronize(device)
            seconds+=time.perf_counter()-start
            losses.append((len(x),float(method.loss(output,y,m))))
            if counterfactual:
                alt=torch.stack([data.tokens[data.alt[i]] for i in ix.tolist()]).to(device)
                wrong=method.decode(net(x,alt),topk=1).detach().float().cpu().numpy()[:,0]
            boxes=boxes.detach().float().cpu().numpy()
            for j,i in enumerate(ix.tolist()):
                row=data.rows[i]
                gt=normalized(row['grasps'],row.get('width',416),row.get('height',416))
                good=correct(boxes[j,0],gt)
                top1+=good
                top5+=good or any(correct(b,gt) for b in boxes[j,1:])
                count+=1
                if counterfactual:
                    wrong_good=correct(wrong[j],gt)
                    shuffled+=wrong_good
                    alt_index=data.same_scene[i]
                    if alt_index is not None:
                        pair_count+=1
                        counterpart=data.rows[alt_index]
                        alt_gt=normalized(counterpart['grasps'],counterpart.get('width',416),counterpart.get('height',416))
                        pair_correct+=good and not wrong_good and correct(wrong[j],alt_gt)
                    predictions.append({'scene_id':row['scene_id'],'prompt':row['prompt'],
                                        'target_id':row['target_id'],'image':row['image'],
                                        'success':bool(good),'gt':gt[0].tolist(),'prediction':boxes[j,0].tolist()})
    results={'n':count,'top1':top1/count,'top5':top5/count,'loss':sum(n*v for n,v in losses)/count,
             'gpu_batch_inference_ms_per_image':1000*seconds/count}
    if counterfactual:
        results.update(shuffled_prompt_top1=shuffled/count,same_scene_other_object_examples=pair_count,
                       same_scene_target_switch_accuracy=pair_correct/pair_count if pair_count else None)
    return results,predictions


def overlay(items,images,path):
    chosen=[r for r in items if r['success']][:6]+[r for r in items if not r['success']][:6]
    tiles=[]
    for row in chosen:
        img=cv2.imread(str(images/row['image']))
        if img is None:
            continue
        h,w=img.shape[:2]
        margin=max(w,h)
        canvas=np.zeros((margin,margin,3),dtype=np.uint8)
        x,y=(margin-w)//2,(margin-h)//2
        canvas[y:y+h,x:x+w]=img
        canvas=cv2.resize(canvas,(256,256))
        for key,col in [('gt',(50,220,75)),('prediction',(30,110,255))]:
            shape=cv2.boxPoints(cv_rect(row[key]))*256/SIZE
            cv2.polylines(canvas,[shape.astype(np.int32)],True,col,2,cv2.LINE_AA)
        canvas=cv2.copyMakeBorder(canvas,0,37,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
        for j in range(2):
            part=row['prompt'][j*39:(j+1)*39].encode('ascii','replace').decode()
            cv2.putText(canvas,part,(7,270+j*15),cv2.FONT_HERSHEY_SIMPLEX,.37,(230,230,230),1,cv2.LINE_AA)
        tiles.append(canvas)
    if tiles:
        while len(tiles)%3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(path),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))


def main():
    a=arguments()
    if a.epochs<1 or a.batch<1:
        raise ValueError('Epochs and batch size must be positive')
    a.output.mkdir(exist_ok=False, parents=True, mode=0o700)
    seed=a.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    cv2.setNumThreads(0)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expected one CUDA GPU selected by the launcher')
    device=torch.device('cuda:0')
    datasets={split:read_rows(a.splits/(split+'.jsonl')) for split in ('train','val','test')}
    sets={k:{r['scene_id'] for r in vals} for k,vals in datasets.items()}
    assert all(sets.values()) and not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    vocabulary=vocab_for(datasets['train'])
    safe_json(a.output/'vocabulary.json',vocabulary)
    config=vars(a)|{'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(device),
                   'image_size':SIZE,'max_targets':MAX_TARGETS,'data_sizes':{k:len(r) for k,r in datasets.items()},
                   'logical_gpu':__import__('os').environ.get('CUDA_VISIBLE_DEVICES','')}
    safe_json(a.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in config.items()})
    loaders={split:DataLoader(GraspDataset(rows,a.images,vocabulary,split=='train'),batch_size=a.batch,
                             shuffle=split=='train',num_workers=a.workers,pin_memory=True,
                             persistent_workers=a.workers>0,generator=torch.Generator().manual_seed(seed))
             for split,rows in datasets.items()}
    method=load_method(a.method)
    model=method.build(len(vocabulary)+3).to(device)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=0.08*a.lr)
    history=[]
    best=(-1.,-1.,-math.inf)
    begun=time.perf_counter()
    weights=a.output/'best.pt'
    for epoch in range(1,a.epochs+1):
        model.train()
        running=0.
        seen=0
        for image,text,target,valid,_ in loaders['train']:
            image,text,target,valid=(v.to(device,non_blocking=True) for v in (image,text,target,valid))
            optimizer.zero_grad(set_to_none=True)
            loss=method.loss(model(image,text),target,valid)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True)
            optimizer.step()
            seen+=len(image)
            running+=float(loss.detach())*len(image)
        val,_=evaluate(method,model,loaders['val'],device)
        record={'epoch':epoch,'train_loss':running/seen,'validation':val,'elapsed_s':time.perf_counter()-begun}
        history.append(record)
        score=(val['top1'],val['top5'],-val['loss'])
        if score>best:
            best=score
            temp=weights.with_suffix('.tmp')
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':val},temp)
            temp.replace(weights)
        print(json.dumps(record),flush=True)
        safe_json(a.output/'history.json',history)
        scheduler.step()
    cp=torch.load(weights,map_location=device,weights_only=True)
    model.load_state_dict(cp['model'])
    testing,items=evaluate(method,model,loaders['test'],device,counterfactual=True)
    report={'best_epoch':cp['epoch'],'best_validation':cp['validation'],'test':testing,'trainable_parameters':count,
            'epochs_completed':a.epochs,'training_and_eval_seconds':time.perf_counter()-begun}
    with (a.output/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in items:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    overlay(items,a.images,a.output/'qualitative.jpg')
    safe_json(a.output/'metrics.json',report)
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
