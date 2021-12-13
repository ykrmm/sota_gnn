



def batch(sample, n=1):
    """Return three batches, a batch of user, batch of positive items, and a batch of negative items

    Args:
        sample (tuple): sample of negative sampling (u,i,j). With u the users,i the positive items, and j the negative ones.
        
        n (int, optional): Batch size. Defaults to 1.

    Yields:
        tuple: Batch of user,pos,neg
    """
    l = len(sample[0])
    for ndx in range(0, l, n):
        yield (sample[0][ndx:min(ndx + n, l)],sample[1][ndx:min(ndx + n, l)],sample[2][ndx:min(ndx + n, l)])
