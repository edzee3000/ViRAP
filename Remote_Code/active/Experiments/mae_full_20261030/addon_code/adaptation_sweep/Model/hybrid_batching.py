"""Exact native-shape batching; no content crop, padding, or aspect warping."""
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def target_shape(row,long_edge):
    width,height=row['width'],row['height'];scale=min(1.,long_edge/max(width,height))
    return max(1,round(height*scale)),max(1,round(width*scale))


class ExactShapeBatches(Sampler):
    def __init__(self,records,long_edge=384,batch_size=16,epochs=4,seed=17,start_batch=0):
        if min(batch_size,epochs,long_edge)<1:raise ValueError('Positive batching dimensions required')
        groups=defaultdict(list)
        for index,row in enumerate(records):groups[target_shape(row,long_edge)].append(index)
        self.batches=[];self.epoch_ends=[]
        for epoch in range(epochs):
            rng=np.random.default_rng(seed+1000+epoch);chunks=[]
            for shape in sorted(groups):
                indices=rng.permutation(groups[shape]).tolist()
                chunks.extend(indices[start:start+batch_size] for start in range(0,len(indices),batch_size))
            rng.shuffle(chunks);self.batches.extend(chunks);self.epoch_ends.append(len(self.batches))
        self.start_batch=start_batch
    def __iter__(self):return iter(self.batches[self.start_batch:])
    def __len__(self):return len(self.batches)-self.start_batch


def collate_native(items):
    return {'content':torch.stack([item['content'] for item in items]),'target':torch.stack([item['target'] for item in items]),
            'references':[item['reference'] for item in items],'pair_ids':[item['pair_id'] for item in items]}


def gpu_batch(batch,device,start=0,end=None):
    end=len(batch['references']) if end is None else end
    # Match the existing style encoder's bilinear+antialias 224 preprocessing
    # on each reference. Its second resize 224->224 is identity.
    references=[F.interpolate(image[None].to(device,non_blocking=True),(224,224),mode='bilinear',align_corners=False,antialias=True)
                for image in batch['references'][start:end]]
    return {'content':batch['content'][start:end].to(device,non_blocking=True),
            'reference':torch.cat(references,0),'target':batch['target'][start:end].to(device,non_blocking=True)}
