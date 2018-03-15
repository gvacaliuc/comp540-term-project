"""
submission.py

Contains methods to process nuclei and individual nucleus predictions by
performing the requisite RLE encoding.
"""

def encode_rle_single_mask(nucleus_mask):
    """
    Run length encodes a single mask.

    :param nucleus_mask: a ndarray masking a single nucleus in the image
    :return encoding: a list of tuples holding the run length encoding of a 
                      mask
    """
