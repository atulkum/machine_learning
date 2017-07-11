from HMM_mine import *

# We experiment with joint probabilities first

hmm = HMM_mine({"FF":0.9, "FL":0.1, "LF":0.1, "LL":0.9}, # transition matrix A
          {"FH":0.5, "FT":0.5, "LH":0.75, "LT":0.25}, # emission matrix E
          {"F":0.5, "L":0.5}) # initial probabilities I
jprob1 = hmm.jointProb("FFFLLLFFFFF", "THTHHHTHTTH")
myprob1 = (0.5 ** 9) * (0.75 ** 3) * (0.9 ** 8) * (0.1 ** 2)
print jprob1, myprob1
# these should be about equal

# confirming that log version of jointProb works as expected
jprobL1 = hmm.jointProbL("FFFLLLFFFFF", "THTHHHTHTTH")
print math.log(jprob1, 2), jprobL1

# Trying another path
jprob2 = hmm.jointProb("FFFFFFFFFFF", "THTHHHTHTTH")
myprob2 = (0.5 ** 12) * (0.9 ** 10)
print jprob2, myprob2
# these should be about equal

# Note that jprob2 is greater than jprob1

# Now we experiment with viterbi decoding
jprobOpt, path = hmm.viterbi("THTHHHTHTTH")
print path

# maximum likelihood path is same path (all fair) as the second one
# we tried above, so jprobOpt should equal jprob2
print jprobOpt, jprob2

# confirming that log version of viterbi works as expected
jprobLOpt, _ = hmm.viterbiL("THTHHHTHTTH")
print math.log(jprobOpt, 2), jprobLOpt

# Now let's make a new HMM with the same states but where jumps
# between fair (F) and loaded (L) are much more probable
hmm = HMM_mine({"FF":0.6, "FL":0.4, "LF":0.4, "LL":0.6}, # transition matrix A
          {"FH":0.5, "FT":0.5, "LH":0.8, "LT":0.2}, # emission matrix E
          {"F":0.5, "L":0.5}) # initial probabilities I
print hmm.viterbi("THTHHHTHTTH")

# Here's an example of underflow.  Note that probability returned
# is 0.0 and the state string becomes all Fs after a while.
print hmm.viterbi("THTHHHTHTTH" * 100)

# Moving to log2 domain fixes underflow
print hmm.viterbiL("THTHHHTHTTH" * 100)

cpgHmm = HMM_mine({'IO':0.20, 'OI':0.20, 'II':0.80, 'OO':0.80},
             {'IA':0.10, 'IC':0.40, 'IG':0.40, 'IT':0.10,
              'OA':0.25, 'OC':0.25, 'OG':0.25, 'OT':0.25},
             {'I' :0.50, 'O' :0.50})
x = 'ATATATACGCGCGCGCGCGCGATATATATATATA'
logp, path = cpgHmm.viterbiL(x)
print x
print path # finds the CpG island fine

x = 'ATATCGCGCGCGATATATCGCGCGCGATATATAT'
logp, path = cpgHmm.viterbiL(x)
print x
print path # finds two CpG islands fine

x = 'ATATATACCCCCCCCCCCCCCATATATATATATA'
logp, path = cpgHmm.viterbiL(x)
print x
print path # oops! - this is just a bunch of Cs.  What went wrong?

