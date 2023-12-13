import datasampler.class_random_sampler
import datasampler.random_sampler
import datasampler.greedy_coreset_sampler
import datasampler.fid_batchmatch_sampler
import datasampler.disthist_batchmatch_sampler
import datasampler.d2_coreset_sampler
import datasampler.lt_sampler
import datasampler.over_sampler

def select(sampler, opt, image_dict, image_list=None, sampling_ratio = None, **kwargs):
    if 'batchmatch' in sampler:
        if sampler=='disthist_batchmatch':
            sampler_lib = disthist_batchmatch_sampler
        elif sampler=='fid_batchmatch':
            sampler_lib = spc_fid_batchmatch_sampler
    elif 'random' in sampler:
        if 'class' in sampler:
            sampler_lib = class_random_sampler
        elif 'full' in sampler:
            sampler_lib = random_sampler
    elif 'coreset' in sampler:
        if 'greedy' in sampler:
            sampler_lib = greedy_coreset_sampler
        elif 'd2' in sampler:
            sampler_lib = d2_coreset_sampler
    elif 'lt' in sampler:
        return lt_sampler.Sampler(opt, sampling_ratio = sampling_ratio, image_dict=image_dict,image_list=image_list)
        
    elif 'over' in sampler:
        sampler_lib = over_sampler    
    else:
        raise Exception('Minibatch sampler <{}> not available!'.format(sampler))

    sampler = sampler_lib.Sampler(opt,image_dict=image_dict,image_list=image_list)

    return sampler
