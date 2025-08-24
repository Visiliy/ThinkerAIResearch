from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt


tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny', use_fast=True)
model = AutoModel.from_pretrained('cointegrated/rubert-tiny')

text = "По словам Дмитрия Донского, в 1998 году он был первым человеком, который смог"

inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=True)

model_inputs = {k: v for k, v in inputs.items() if k != 'offset_mapping'}
with torch.no_grad():
    outputs = model(**model_inputs)

last_hidden_states = outputs.last_hidden_state.squeeze(0)
seq, embed_dim = last_hidden_states.shape
vector = last_hidden_states.flatten()

fft_vector = torch.fft.fft(vector)
amplitude_spectrum = torch.abs(fft_vector).cpu().numpy()
result = torch.tensor(amplitude_spectrum.reshape((seq, embed_dim)))
print(result)
plt.figure(figsize=(12, 6))
plt.plot(amplitude_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
plt.show()
