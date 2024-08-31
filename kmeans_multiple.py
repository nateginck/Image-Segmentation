from kmeans_single import kmeans_single

def kmeans_multiple(X, K, iters, R):
    # create empty list for values
    ssd = []
    ids = []
    means =[]

    # iterate through R
    for i in range(R):
        current_ids, current_means, current_ssd = kmeans_single(X, K, iters)

        # initalize values based on first run
        if i == 0:
            ssd.append(current_ssd)
            ids.append(current_ids)
            means.append(current_means)

        # check if ssd is lower
        else:
            if ssd[i-1] > current_ssd:
                ssd[i-1] = current_ssd
                ids[i-1] = current_ids
                means[i-1] = current_means

        # return values
        return ids[0], means[0], ssd[0]
