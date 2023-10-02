import numpyro

def check_rhat(model):
    grouped_samples = model.get_samples(group_by_chain=True)
    
    max_rhat = 0
    for idx, samples in grouped_samples.items():
        temp_rhat = numpyro.diagnostics.gelman_rubin(samples)
        if temp_rhat > max_rhat:
            max_rhat = temp_rhat
            
    return max_rhat

def check_ess(model):
    grouped_samples = model.get_samples(group_by_chain=True)
    grouped_samples = {k: v for k, v in grouped_samples.items() if k[0] == 'q'}
    
    min_ess = 10e9
    for idx, samples in grouped_samples.items():
        temp_ess = numpyro.diagnostics.effective_sample_size(samples)
        if temp_ess < min_ess:
            min_ess = temp_ess
            
    return min_ess