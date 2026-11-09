"""Offline saved-action, normalization, conditioning and cache-parity diagnostic. No training/hardware."""
import json
from pathlib import Path
import numpy as np
from pi05_train_preflight import build_config
import torch
import pyarrow.parquet as pq
from lerobot.policies.pi05.prompt import QuantileNormalizer, QuantileStats, Pi05PromptBuilder

RUN=Path('outputs/train/pi05_lora_stages_20261105_162719')
OUT=RUN/'action_diagnostic'
OUT.mkdir(exist_ok=True)
z=np.load(RUN/'trajectories.npz')
r=json.loads((RUN/'report.json').read_text())
root=Path('D:/1104/nero_cube_300_pi05')
norm=QuantileNormalizer(QuantileStats.from_json(root/'train/meta/quantile_stats.json'))
prompt=Pi05PromptBuilder(str(Path('outputs/model/tokenizer').resolve()),200,32)
result={'hardware_access':False,'training_started':False,'saved_actions':{},'data':{}}

def save():
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')

def crossing(g,close):
    mask=g<=.5 if close else g>=.5
    return next((i for i in range(len(g)-2) if mask[i:i+3].all()),None)

def timing(pred,target,choices):
    output={}
    for stage in ('grasp','release'):
        trials=initial=found=0;lags=[]
        for i,x in enumerate(choices):
            if x['stage']!=stage:continue
            close=stage=='grasp';ref=crossing(target[i,:,7],close)
            for p in pred:
                trials+=1
                ok=p[i,0,7]>.5 if close else p[i,0,7]<.5
                initial+=int(ok);hit=crossing(p[i,:,7],close) if ok else None
                if hit is not None and ref is not None:
                    found+=1;lags.append(hit-ref)
        output[stage]={'trials':trials,'initial_side_correct':initial,'valid_crossings':found,
                       'lag_frames':lags,'median_lag_seconds':float(np.median(lags)/30) if lags else None}
    return output

data={}
schedule=json.loads((RUN/'sampling_manifest.json').read_text())['training']
for split in ('train','val'):
    aa=[];ss=[];bags={}
    for path in sorted((root/split/'data').glob('*/*.parquet')):
        t=pq.read_table(path,columns=['action','observation.state','episode_index'])
        a=np.asarray(t['action'].to_pylist(),np.float32);s=np.asarray(t['observation.state'].to_pylist(),np.float32)
        ep=int(t['episode_index'][0].as_py());data[split,ep]=(s,a);aa.append(a);ss.append(s)
        bags[ep]=str(tuple(float(v) for v in np.round(np.median(s[:,13:15],axis=0),1)))
    a=np.concatenate(aa);s=np.concatenate(ss)
    roundtrip=norm.inverse_action(norm.action(torch.from_numpy(a))).numpy()
    counts={};sampled={}
    for key in bags.values():counts[key]=counts.get(key,0)+1
    if split=='train':
        for x in schedule:
            key=bags[x['episode']];sampled[key]=sampled.get(key,0)+1
    result['data'][split]={'episodes':len(bags),'frames':len(a),
        'action_vs_state_per_joint_MAE_deg':(abs(a[:,:7]-s[:,:7]).mean(0)*180/np.pi).tolist(),
        'action_vs_state_gripper_MAE':float(abs(a[:,7]-s[:,7]).mean()),
        'roundtrip_max_abs':float(abs(roundtrip-a).max()),
        'bag_rounded_episode_median_counts':counts,'training_windows_by_bag_proxy':sampled,
        'bag_note':'Rounded image-space episode medians are approximate proxies, not physical placement labels.'}
for panel,choices in r['selected'].items():
    target=z[panel+'_target'];state=z[panel+'_hold'][:,0]
    split='train' if panel=='train_probe' else 'val';states=[];maxdiff=0.
    for i,x in enumerate(choices):
        s,a=data[split,x['episode']];f=x['frame'];states.append(s[f])
        maxdiff=max(maxdiff,float(abs(target[i]-a[f:f+50]).max()))
    states=torch.from_numpy(np.stack(states))
    texts=prompt.build_prompts(['Pick up the red cube and put it into the bag.']*len(states),norm.state(states))
    lens=[len(v) for v in prompt.tokenizer(texts,truncation=False)['input_ids']]
    result['saved_actions'][panel]={'target0_vs_state_joint_MAE_deg':float(abs(target[:,0,:7]-state[:,:7]).mean()*180/np.pi),
        'saved_target_vs_parquet_max_abs':maxdiff,'prompt_token_min':min(lens),'prompt_token_max':max(lens),'truncated_count':sum(n>200 for n in lens)}
    for step in (0,200,400,800):
        if f'{panel}_step_{step}_seed_31415' not in z:continue
        pred=np.stack([z[f'{panel}_step_{step}_seed_{seed}'] for seed in (31415,27182)])
        diff=pred-target[None]
        anchored=(pred-pred[:,:,:1])-(target[None]-target[None,:,:1])
        result['saved_actions'][panel][str(step)]={
            'first_joint_MAE_deg':float(abs(diff[:,:,0,:7]).mean()*180/np.pi),
            'first_per_joint_MAE_deg':(abs(diff[:,:,0,:7]).mean((0,1))*180/np.pi).tolist(),
            'first_per_joint_bias_deg':(diff[:,:,0,:7].mean((0,1))*180/np.pi).tolist(),
            'h50_per_joint_MAE_deg':(abs(diff[...,:7]).mean((0,1,2))*180/np.pi).tolist(),
            'h50_joint_MAE_deg':float(abs(diff[...,:7]).mean()*180/np.pi),
            'first_jump_from_state_deg':float(abs(pred[:,:,0,:7]-state[None,:,:7]).mean()*180/np.pi),
            'shape_only_joint_MAE_deg_oracle_anchor':float(abs(anchored[...,:7]).mean()*180/np.pi),
            'first_gripper_MAE':float(abs(diff[:,:,0,7]).mean()),
            'h50_gripper_MAE':float(abs(diff[...,7]).mean()),
            'seed_joint_difference_deg':float(abs(pred[0,...,:7]-pred[1,...,:7]).mean()*180/np.pi),
            'gripper_outside_fraction':float(((pred[...,7]<0)|(pred[...,7]>1)).mean()),
            'gripper_timing':timing(pred,target,choices)}
save()
print('CPU_DIAGNOSTIC_PASS',flush=True)

if __name__=='__main__' and '--gpu' in __import__('sys').argv:
    from pi05_lora_short_train import make_dataset,get_batch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.common.vla_utils import make_att_2d_masks,prepare_attention_masks_4d
    from safetensors.torch import load_file
    config,info=build_config(Path('outputs/model'),root/'train',Path('outputs/model/tokenizer'),'bfloat16')
    dataset=make_dataset(root/'val','val',info['fps'])
    choices=r['selected']['validation_stages'][:8]
    batches=[get_batch(dataset,x['dataset_index']) for x in choices]
    policy=PI05Policy.from_pretrained('outputs/model',config=config,strict=True,local_files_only=True)
    result['gpu_probe']={'samples':choices,'noise_seed':31415,'inference_steps':10,
        'note':'8 windows from 2 episodes. Sensitivity is not proof of correct use. Oracle anchoring is diagnostic only.'}
    for label,step in [('base',0),('best400',400)]:
        if step:
            weights=load_file(RUN/'checkpoint_best/model.safetensors')
            loaded=policy.load_state_dict(weights,strict=False)
            expected={n for n,p in policy.named_parameters() if p.requires_grad}
            if set(weights)!=expected or loaded.unexpected_keys:raise RuntimeError('Adapter mismatch')
        policy.eval();predictions={k:[] for k in ('correct','robot_state_midpoint','env_swapped','images_zero')};parity=[]
        with torch.no_grad():
            for i,batch in enumerate(batches):
                obs={k:batch[k] for k in ('observation.state','observation.images.base','observation.images.wrist','task')}
                noise=torch.randn((1,50,32),generator=torch.Generator(device='cuda').manual_seed(31415+i),device='cuda')
                for condition in predictions:
                    modified=dict(obs)
                    if condition=='robot_state_midpoint':
                        modified['observation.state']=obs['observation.state'].clone()
                        modified['observation.state'][:,:8]=(norm.state_q01[:8]+norm.state_q99[:8]).to('cuda')/2
                    elif condition=='env_swapped':
                        modified['observation.state']=obs['observation.state'].clone()
                        modified['observation.state'][:,8:]=batches[(i+4)%8]['observation.state'][:,8:]
                    elif condition=='images_zero':
                        for k in ('observation.images.base','observation.images.wrist'):modified[k]=torch.zeros_like(obs[k])
                    predictions[condition].append(policy.predict_action_chunk(modified,noise=noise,num_steps=10)[0].cpu().numpy())
                if i in (0,3):
                    images,img_masks=policy._preprocess_images(obs);tokens,masks=policy._prepare_language(obs)
                    actions=policy.prepare_action(batch);time=torch.tensor([.5],device='cuda');model=policy.model
                    full=model(images,img_masks,tokens,masks,actions,noise,time)
                    prefix,pm,am=model.embed_prefix(images,img_masks,tokens,masks)
                    _,cache=model.paligemma_with_expert.forward(
                        attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pm,am)),
                        position_ids=torch.cumsum(pm,dim=1)-1,past_key_values=None,
                        inputs_embeds=[prefix,None],use_cache=True)
                    velocity=model.denoise_step(pm,cache,.5*noise+.5*actions,time)
                    cached=((noise-actions)-velocity)**2;delta=(cached[:,:,:8]-full[:,:,:8]).abs()
                    parity.append({'sample':i,'full_loss':float(full[:,:,:8].mean()),'cached_loss':float(cached[:,:,:8].mean()),
                        'loss_tensor_max_abs_difference':float(delta.max()),'loss_tensor_mean_abs_difference':float(delta.mean())})
                print(f'PROBE {label} {i+1}/8',flush=True)
        predictions={k:np.stack(v) for k,v in predictions.items()}
        for v in predictions.values():
            if not np.isfinite(v).all():raise RuntimeError('Nonfinite probe prediction')
        np.savez_compressed(OUT/f'{label}_conditioning.npz',**predictions)
        correct=predictions['correct'];effects={}
        for condition,pred in predictions.items():
            d=pred-correct
            effects[condition]={'joint_change_deg':float(abs(d[...,:7]).mean()*180/np.pi),
                'first_joint_change_deg':float(abs(d[:,0,:7]).mean()*180/np.pi),
                'gripper_change':float(abs(d[...,7]).mean()),
                'joint_MAE_deg':float(abs(pred[...,:7]-z['validation_stages_target'][:8,:,:7]).mean()*180/np.pi)}
        saved=z[f'validation_stages_step_{step}_seed_31415'][:8]
        result['gpu_probe'][label]={'conditioning':effects,'full_vs_cached':parity,
            'saved_prediction_max_abs_difference':float(abs(correct-saved).max())}
        save()
    print('GPU_DIAGNOSTIC_PASS',flush=True)
