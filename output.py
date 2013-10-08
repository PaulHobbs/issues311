def output(ids, ys, output_file='submission'):
    """
    Takes the output of several regressors, labels the rows with
    the correct id and writes to a csv file.
    """
    with open('%s.csv' % output_file, 'w') as fp:
        fp.write("id,num_views,num_votes,num_comments")
        for i in range(len(ids)):
            fp.write(",".join((
                ids[i],
                ys['num_views'],
                ys['num_votes'],
                ys['num_comments']
            )))