"""Synthetic architecture diagnostics, not a training/quality benchmark.

Uses explicit RANDOM official DINOv2 structure on CPU. Does not access Trinity,
public pretrained weights, source photographs, or any held-out evaluation data.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tone_model.v2 import ReferenceToneModelV2


def maximum_gradient(module):
    values=[p.grad.abs().max().item() for p in module.parameters() if p.grad is not None]
    return max(values,default=0.)


def main():
    torch.set_num_threads(4);torch.manual_seed(20261030)
    repo=Path(__file__).resolve().parents[1]/'Research/External_Code/dinov2'
    model=ReferenceToneModelV2.from_official(repo,allow_uninitialized=True)
    image=torch.rand(1,3,37,49);reference=torch.rand(1,3,43,31);target=(image*.8+.1).clamp(0,1)
    model.set_stage('encoder');model.zero_grad(set_to_none=True)
    prediction,aux=model(image,reference,True);F.l1_loss(prediction,target).backward()
    report={'scope':'Untrained random structure, synthetic CPU diagnostics only',
            'encoder_only_from_identity_initialization':{
                'reconstruction_loss':F.l1_loss(prediction,target).item(),
                'maximum_encoder_gradient':maximum_gradient(model.encoder),
                'reason':'Frozen zero-initialized color head makes the rendered image independent of encoder parameters at initialization.'},
            'missing_v1_loss_aux_keys':sorted({'global_delta','preclamp_output'}-aux.keys())}
    model.set_stage('heads');optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    trace=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction=model(image,reference);F.l1_loss(prediction,target).backward()
        trace.append({'step':step+1,'encoder_gradient':maximum_gradient(model.encoder),
                      'conditioner_gradient':maximum_gradient(model.conditioner),
                      'color_output_gradient':maximum_gradient(model.color_generator.output),
                      'color_film_gradient':maximum_gradient(model.color_generator.film),
                      'spatial_film_gradient':maximum_gradient(model.content_context.pair_film)})
        optimizer.step()
    report['joint_heads_initial_gradient_trace']=trace
    # Four same-color source locations with an interior RGB tetrahedron target.
    # An arbitrary global LUT contributes one common RGB translation. K=2
    # local color vectors plus arbitrary maps still lie in an affine plane.
    desired=torch.tensor([[.4,.4,.4],[.5,.4,.4],[.4,.5,.4],[.4,.4,.5]],dtype=torch.float64)
    centered=desired-desired.mean(0,keepdim=True)
    singular=torch.linalg.svdvals(centered)
    report['same_input_color_rank_counterexample']={
        'desired_region_rgb':desired.tolist(),'centered_target_rank':int(torch.linalg.matrix_rank(centered)),
        'centered_singular_values':singular.tolist(),
        'rank2_preclamp_best_affine_plane_rgb_rmse':float((singular[2:].square().sum()/desired.numel()).sqrt()),
        'interpretation':'Even arbitrary reference-conditioned maps cannot fit every same-color regional RGB edit with K=2. This is not an error bound for clipped outputs.'}
    raw=torch.tensor([-.2,.5,1.2],requires_grad=True)
    F.l1_loss(raw.clamp(0,1),torch.full_like(raw,.4)).backward()
    report['clamped_reconstruction_gradient']={'raw':raw.detach().tolist(),'gradient':raw.grad.tolist()}
    cfg=model.config
    report['fixed_limits']={'context_grid':cfg.context_size,'local_rank':cfg.local_rank,
        'local_output_absolute_bound_default_none':cfg.local_output_limit,
        'initial_gate_bound_not_initial_actual_output':cfg.local_limit*cfg.local_rank*torch.sigmoid(torch.tensor(-2.)).item(),
        'token_count_prior_if_logits_equal':{'visual':1203/1399,'photometric_cnn':192/1399,'statistics':4/1399}}
    # A legal parameter setting suffices to disprove a hard architectural
    # self-reference identity guarantee; this is not a learned-model evaluation.
    with torch.no_grad():
        model.color_generator.output.weight.zero_();model.color_generator.output.bias.fill_(.2)
        output=model(image,image)
    report['self_reference_identity_not_guaranteed']={
        'legal_manual_color_head_bias':.2,'mean_absolute_change':(output-image).abs().mean().item()}
    path=Path(__file__).parent/'v2_1_audit_results.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
