"""Bounded real-weight full/cached velocity and FP32 numerical control. Offline only."""
import argparse
import gc
import json
from pathlib import Path
from pi05_train_preflight import build_config
from pi05_lora_short_train import make_dataset,get_batch
import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.common.vla_utils import make_att_2d_masks,prepare_attention_masks_4d

def difference(a,b):
    d=(a-b).abs()[...,:8]
    return {'mean_abs':float(d.mean()),'max_abs':float(d.max()),
            'rms':float(d.square().mean().sqrt()),
            'reference_rms':float(b[...,:8].square().mean().sqrt())}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());root=Path(manifest['dataset'])
    config,info=build_config(Path('outputs/model'),root/'train',Path('outputs/model/tokenizer'),'bfloat16')
    dataset=make_dataset(root/'val','val',info['fps'])
    # First and last occupied cells, with a complete grasp window in each.
    eligible=[x for x in manifest['evaluation']['validation_stages'] if x['stage']=='grasp']
    choices=[eligible[0],eligible[-1]]
    batches=[get_batch(dataset,x['dataset_index']) for x in choices]
    torch.manual_seed(42)
    policy=PI05Policy.from_pretrained('outputs/model',config=config,strict=True,local_files_only=True)
    policy.eval()
    # Match sample_actions/denoise_step; config defaults can otherwise select SDPA.
    policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation='eager'
    policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation='eager'
    result={'hardware_access':False,'training_started':False,'samples':choices,
        'note':'FP32 uses the same BF16-rounded stored weights converted to FP32; this isolates arithmetic, not weight quantization. State interventions are cross-modal inconsistent sensitivity probes.',
        'results':{}};arrays={}
    def save():
        (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        np.savez_compressed(a.output/'trajectories.npz',**arrays)
    with torch.no_grad():
        # BF16 base and prior trained adapter first, then FP32 prior adapter.
        for label in ('base_bf16','best400_bf16','best400_fp32'):
            if label=='best400_bf16':
                weights=load_file('outputs/train/pi05_lora_stages_20261105_162719/checkpoint_best/model.safetensors')
                expected={n for n,p in policy.named_parameters() if p.requires_grad}
                if set(weights)!=expected:raise RuntimeError('Adapter keys mismatch')
                policy.load_state_dict(weights,strict=False);del weights
            if label.endswith('fp32'):
                gc.collect();torch.cuda.empty_cache()
                policy.model.paligemma_with_expert.to(torch.float32)
            result['results'][label]=[]
            for i,batch in enumerate(batches):
                obs={k:batch[k] for k in ('observation.state','observation.images.base','observation.images.wrist','task')}
                images,im=policy._preprocess_images(obs);tokens,masks=policy._prepare_language(obs)
                model=policy.model;actions=policy.prepare_action(batch)
                noise=torch.randn((1,50,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(31415+i))
                prefix,pm,am=model.embed_prefix(images,im,tokens,masks)
                _,cache=model.paligemma_with_expert.forward(attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pm,am)),
                    position_ids=torch.cumsum(pm,dim=1)-1,past_key_values=None,inputs_embeds=[prefix,None],use_cache=True)
                captured=[]
                def hook(module,args,output):captured.append(output.detach())
                handle=model.action_out_proj.register_forward_hook(hook)
                def full(x,t):
                    # Choose actions=x and noise=x so actual training forward sees x at every t.
                    captured.clear();model(images,im,tokens,masks,x,x,t)
                    if len(captured)!=1:raise RuntimeError('Unexpected forward hook count')
                    return captured[0].clone()
                parity=[]
                try:
                    for v in (1.,.5,.1):
                        t=torch.tensor([v],device='cuda');x=v*noise+(1-v)*actions
                        vf=full(x,t);vc=model.denoise_step(pm,cache,x,t)
                        parity.append({'time':v,**difference(vc,vf)})
                    xf=noise.clone();xc=noise.clone()
                    for j in range(10):
                        t=torch.tensor([1-j/10],device='cuda')
                        xf=xf-.1*full(xf,t);xc=xc-.1*model.denoise_step(pm,cache,xc,t)
                finally:handle.remove()
                # Compare normalized and inverse-normalized full Euler trajectories.
                rawf=policy.quantile_normalizer.inverse_action(xf[...,:8]);rawc=policy.quantile_normalizer.inverse_action(xc[...,:8])
                arrays[label+f'_{i}_full']=rawf.cpu().numpy();arrays[label+f'_{i}_cached']=rawc.cpu().numpy()
                entry={'velocity_full_vs_cached':parity,'euler_normalized':difference(xc,xf),
                       'euler_joint_difference_deg':float((rawc[...,:7]-rawf[...,:7]).abs().mean()*180/np.pi),
                       'euler_gripper_difference':float((rawc[...,7]-rawf[...,7]).abs().mean())}
                # Matched noise, real state swap across separated bag-position episodes.
                correct=policy.predict_action_chunk(obs,noise=noise,num_steps=10)
                interventions={}
                for name,sl in [('joints',slice(0,7)),('bag_xy',slice(13,15))]:
                    modified=dict(obs);modified['observation.state']=obs['observation.state'].clone()
                    modified['observation.state'][:,sl]=batches[1-i]['observation.state'][:,sl]
                    other_tokens,_=policy._prepare_language(modified)
                    pred=policy.predict_action_chunk(modified,noise=noise,num_steps=10)
                    interventions[name]={'input_mean_abs':float((modified['observation.state'][:,sl]-obs['observation.state'][:,sl]).abs().mean()),
                        'changed_tokens':int((other_tokens!=tokens).sum()),
                        'output_joint_change_deg':float((pred[...,:7]-correct[...,:7]).abs().mean()*180/np.pi),
                        'output_gripper_change':float((pred[...,7]-correct[...,7]).abs().mean())}
                entry['interventions']=interventions
                entry['cached_predict_api_max_difference']=float((rawc-correct).abs().max())
                result['results'][label].append(entry);save();print(label,i,json.dumps(entry),flush=True)
                del cache,prefix,captured,xf,xc,vf,vc,rawf,rawc,pred,correct
            if label.endswith('fp32'):
                result['fp32_vs_bf16_cached_joint_deg']=[float(np.abs(arrays[f'best400_fp32_{i}_cached'][...,:7]-arrays[f'best400_bf16_{i}_cached'][...,:7]).mean()*180/np.pi) for i in range(2)]
            save()
    result['status']='PASS';save();print('NUMERICAL_PROBE_PASS',flush=True)

if __name__=='__main__':main()
