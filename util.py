def entropy(p,n):
    p_ratio = float(p)/(p+n)
    n_ratio = float(n)/(p+n)
    return -p_ratio*math.log(p_ratio) - n_ratio * math.log(n_ratio)

def info_gain(p0,n0,p1,n1,p,n):
    return entropy(p,n) - float(p0+n0)/(p+n)*entropy(p0,n0) - float(p1+n1)/(p+n)*entropy(p1,n1)



def Heap_gain(p, n, class_label, dict_all, num_features = 1000, gain_minimum_bar = -100):
    heap = [(gain_minimum_bar, 'gain_bar')] * num_features
    root = heap[0]
    for gram, count_list in dict_all.iteritems():
        p1 = count_list[class_label-1]
        n1 = sum(count_list[:(class_label-1)] + count_list[class_label:])
        p0,n0 = p - p1, n - n1
        if p1*p0*n1*n0 != 0:
            gain = info_gain(p0,n0,p1,n1,p,n)
            if gain > root[0]:
                root = heapq.heapreplace(heap, (gain, gram))
    #return heap
    result = [i[1] for i in heap if i[1] != 'gain_bar']
    #print "the length of dll for class %i is %i"%(class_label, len(result))
    return result
