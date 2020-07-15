import string
from datasets import dataset
alphabet = string.printable[:-6]
converter = dataset.strLabelToInt(alphabet=alphabet)
data = ['avalislj', 'asdfljaslfj','lkasjf(poajsEjpaoijsdf','lkajfk^&(I()）']
texts,length = converter.encoder(data)
print(converter.decoder(texts,length))





