import string
from datasets import dataset
alphabet = string.printable[:-6]
converter = dataset.strLabelToInt(alphabet=alphabet)
print(converter.voc)
print(converter.dict)



