import struct
import math
import kolmogorov_compl
import numpy as np
import zlib

def bin(i):
    l = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
         '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']
    hex_str = hex(i)[2:]
    if hex_str[-1] == 'L':
        s = ''.join(map(lambda x, l=l: l[int(x, 16)], hex_str[:-1]) )
    else:
        s = ''.join(map(lambda x, l=l: l[int(x, 16)], hex_str) )
#    if s[0] == '1' and i > 0:
#        s = '0000' + s
    return s


def float2bin(num):
    if num == 0:
        return '0' * 64
    i = struct.unpack('!Q', struct.pack("!d", num))[0]
    return bin(i)

def iterable2bin(iterable):
    s = ''
    for i in iterable:
        s = s + float2bin(i)
    return s

#% FUNCTION: kolmogorov.m
#% DATE: 9th Feb 2005
#% AUTHOR: Stephen Faul (stephenf@rennes.ucc.ie)
#%
#% Function for estimating the Kolmogorov Complexity as per:
#% "Easily Calculable Measure for the Complexity of Spatiotemporal Patterns"
#% by F Kaspar and HG Schuster, Physical Review A, vol 36, num 2 pg 842
#%
#% Input is a digital string, so conversion from signal to a digital stream
#% must be carried out a priori

def py_kolmogorov(s):
   n=len(s);
   c=1;
   l=1;

   i=0;
   k=1;
   k_max=1;
   stop = False;

   while not stop:
       if s[i+k -1 ] != s[l+k -1]:
           if k>k_max:
               k_max=k
           i=i+1

           if i==l:
               c=c+1
               l=l+k_max
               if l+1>n:
                   stop=1
               else:
                   i=0
                   k=1
                   k_max=1
           else:
               k=1
       else:
           k=k+1
           if l+k>n:
               c=c+1
               stop=True

   print "n", n
   print "c", c
   b=float(n)/ (math.log(n) / math.log(2.))

   #% a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976),
   #% h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
   #% and h(n) is a normalised measure of complexity.
   complexity=float(c)/float(b)

   return complexity

def kolmogorov_zlib(s):
  compr = zlib.compress(s.tostring())
  return float(len(compr)) / float(len(s.tostring()))
  #n = float(len(s))
  #print "n", n
  #c = float(len(compr))
  #print "c", c
  #b= n/ (math.log(n) / math.log(2.))
  #complexity = c/b
  ##print "b", b
  #return complexity
  

def kolmogorov(s):
    s_arr = np.array([int(x) for x in s],  dtype=np.int)
    #print "starting kolmogorov_compl"
    return  kolmogorov_compl.kolmogorov(s_arr)
    #return py_kolmogorov(s_arr)

def kolmogorov_float(s):
  bin_arr = iterable2bin(s)
  return kolmogorov(bin_arr)

def test_zlib(s):
  print "standard: ", kolmogorov(s)
  print "zlib", kolmogorov_zlib(s)

