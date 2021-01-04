'''
https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
We will convert the InputExample to the feature that is understood by BERT. The feature will be represented by class InputFeatures.
'''

class InputFeatures(object):
    """A single set of features of data.
    - input_ids: list of numerical ids for the tokenised text
    - input_mask: will be set to 1 for real tokens and 0 for the padding tokens
    - segment_ids: for our case, this will be set to the list of ones
    - label_ids: one-hot encoded labels for the text
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids