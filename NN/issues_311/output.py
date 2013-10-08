def output(y, output_file='submission'):
    """
    Takes the output of several regressors, labels the rows with
    the correct id and writes to a csv file.
    """
    with open('%s.csv' % output_file, 'w') as fp:
        fp.write("id,num_views,num_votes,num_comments")
        for row in y:
            fp.write(",".join(row))