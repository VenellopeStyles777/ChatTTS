###################################
# Sample a speaker from Gaussian.
import ChatTTS
from IPython.display import Audio
import scipy
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

chat = ChatTTS.Chat()
chat.load_models()
std, mean = torch.load('models/asset/spk_stat.pt').chunk(2)
rand_spk = torch.randn(768) * std + mean

params_infer_code = {
  'spk_emb': rand_spk, # add sampled speaker 
  'temperature': .3, # using custom temperature
  'top_P': 0.7, # top P decode
  'top_K': 20, # top K decode
}

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
} 

wav = chat.infer("Test Run", params_refine_text=params_refine_text, params_infer_code=params_infer_code)

###################################
# For word level manual control.
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wav = chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
scipy.io.wavfile.write(filename = "./testOutput/3.wav",rate=24_000,data=wav[0].T)
#
