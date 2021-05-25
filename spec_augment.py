import torch
import torchaudio
import torchaudio.transforms as T
import random


class Spec_Augment:
    
    
    def time_stretch(self, spec):
        stretch = T.TimeStretch()
        return stretch(spec, 1.2)
        
    
    def freq_mask(self, spec):
        masking = T.FrequencyMasking(freq_mask_param=80)
        return masking(spec)
    
    def time_mask(self, spec):
        masking = T.TimeMasking(time_mask_param=80)
        return masking(spec)
        
    def main(self, given_spec):
        new_spec = self.time_stretch(given_spec)
        new_spec = self.freq_mask(new_spec)
        final_spec = self.time_mask(new_spec)
        return final_spec