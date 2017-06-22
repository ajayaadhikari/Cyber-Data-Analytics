import pandas as pd

# implementation of FREQUENT (Misra - Gries) algorithm
def frequent(stream, k):
    freqD = {}
    for value in stream:
        if value in freqD:
            freqD[value]+=1

        elif len(freqD) < k-1:
            freqD[value] = 1

        else:
            for existing_value in freqD:
                freqD[existing_value]-=1
            freqD = {x: y for x, y in freqD.items() if y != 0}
        #print(freqD) #(show step-by-step)
    return freqD

if __name__ == '__main__':

    src_ip = open("../data/src_ip.txt", 'r')
    stream_ips = [line[:-2] for line in src_ip.readlines()]

    print(len(set(stream_ips)))
    #data = pd.read_csv("../data/cmds_sequence_2016-07-01.csv")
    #print(list(data))

    # initialize fraction
    k = 10

    # initialize stream
    stream = ["g","g","b","g","b","y","bl","b","g","b","g","b"]

    # find elements whose frequency exceeds 1/k fraction of the total count
    #elements = frequent(stream_ips, k)
    #print("Found %s elements over the total of %s elements" %(len(elements), len(stream_ips)))