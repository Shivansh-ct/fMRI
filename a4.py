import nibabel as nb
import numpy as np
import argparse
from scipy.fftpack import fft, ifft



def slice_time_correction(data, t, slicetimes, TR, f):
    c_img = np.zeros(data.shape)
    if t>TR or t<0:
        f.write("SLICE TIME CORRECTION FAILURE")
        return data
        
    else:
        time = np.arange(0,data.shape[3],1)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    
                        if slicetimes[z]>TR or slicetimes[z]<0:
                            f.write("SLICE TIME CORRECTION FAILURE")
                            return data
                        
                        
                        elif slicetimes[z] > t:
                            alpha = np.array(((t+TR - (slicetimes[z]*np.ones(data.shape[3])))/TR)[0:-1])
                            #print(data[x][y][z][0:-2])
                            c_img[x][y][z][0:-1] = np.add(np.multiply(alpha,data[x][y][z][0:-1]),np.multiply((1-alpha),np.concatenate(([2*data[x][y][z][0]-data[x][y][z][1]],data[x][y][z][0:-2]))))  
                            #print(c_img[x][y][z][1:-1])
                        else:
                            alpha = np.array(((t - (slicetimes[z]*np.ones(data.shape[3])))/TR)[0:-1])
                            c_img[x][y][z][0:-1] =  np.add(np.multiply(alpha,data[x][y][z][1:]),np.multiply((1-alpha),data[x][y][z][0:-1]))                       
                        c_img[x][y][z][data.shape[3]-1] = data[x][y][z][data.shape[3]-1] 
    #c_img[0:][0:][0:][data.shape[3]-1] = data[0:][:][:][data.shape[3]-1]
    return c_img
    
    
    
    
#def temporal_filtering(data, TR, low, high):
    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
    parser.add_argument("-tc", "--slicetime", nargs=2, help="slice time acquisition")
    parser.add_argument("-sm", "--fwhm", help="fwhm")
    parser.add_argument("-tf", "--cutoff", nargs=2, help="temporal filtering")
    parser.add_argument("-ts", "--timeseries", nargs=3, help="time series")
    args = parser.parse_args()
   # print(args.input, args.output, args.cutoff, args.slicetime, args.fwhm)
    
    
    img = nb.load(args.input)
    data = img.get_data()
   # print("hello")
    t = int(args.slicetime[0])
    header = img.header
    TR = header.get_zooms()[3]
    TR = TR*1000
    slicetimes = np.array(np.loadtxt(args.slicetime[1]), dtype=np.float32)
    #print(type(slicetimes[1]))
    f = open(args.output + ".txt", 'a')
    
    
    #print(data.shape, t, slicetimes.shape, TR, type(f))
    
    fMRIData = slice_time_correction(data, t, slicetimes, TR, f)
    print(fMRIData[5][5][5])
    saveNII = True
    outputImg = nb.Nifti1Image(fMRIData, np.eye(4), header=header)
    outputNIIFileName = args.output + '.nii.gz'
    nb.save(outputImg, outputNIIFileName)
    
    

                
