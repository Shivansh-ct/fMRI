import nibabel as nb
import numpy as np
import argparse
from scipy.fftpack import fft, ifft



def temporal_filtering(data, TR, low, high):
    temp_filtered = np.zeros(data.shape)
    if low>high:
        temp = high
        high = low
        low = temp
    high_pass = 1/low
    low_pass = 1/high
    
    freq = np.concatenate((np.linspace(0,1.0/(2.0*TR),data.shape[3]//2),np.linspace(-1.0/(2.0*TR),0,data.shape[3]//2)))
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                timeseries = data[x][y][z][:]
                timeseries = fft(timeseries)
                timeseries[np.abs(freq)>high_pass]=0
                timeseries[np.abs(freq)<low_pass]=0
                timeseries = ifft(timeseries)
                temp_filtered[x][y][z][:] = timeseries
    return temp_filtered



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
    
    
    
    

    
def spatial_smoothing(data, fwhm, kernel_size, voxel_size):
    
    smooth = np.zeros(data.shape)
    for time in range(data.shape[3]):
       #Applying along x direction
       kernel_x = prep_kernel(fwhm, kernel_size, voxel_size[0])
       for y in range(data.shape[1]):
           for z in range(data.shape[2]):
               smooth[:,y,z,time] = apply_kernel(data[:,y,z,time], kernel_x)
       #Applying along y direction
       kernel_y = prep_kernel(fwhm, kernel_size, voxel_size[1])
       for x in range(data.shape[0]):
           for z in range(data.shape[2]):
               smooth[x,:,z,time] = apply_kernel(smooth[x,:,z,time], kernel_y)
       #Applying along z direction
       kernel_z = prep_kernel(fwhm, kernel_size, voxel_size[2])
       for y in range(data.shape[1]):
           for x in range(data.shape[0]):
               smooth[x,y,:,time] = apply_kernel(smooth[x,y,:,time], kernel_z)
    return smooth
    

#Preparing a Gaussian Kernel
def prep_kernel(fwhm, kernel_size, voxel_size):
    v = np.arange(int(-1*kernel_size/2), int(kernel_size/2)+1)
    sigma = fwhm/(np.sqrt(8*np.log(2))*voxel_size)
    w = np.exp(-(v)**2/(2*(sigma**2)))
    kernel = w/sum(w)
    return kernel
    

def apply_kernel(series, kernel):
    
    padded_series = np.concatenate((np.zeros(np.size(kernel)//2), series, np.zeros(np.size(kernel)//2)))
    conv_series = np.zeros(series.shape)
    for i in range(series.shape[0]):
        conv_series[i] = np.sum(np.multiply(padded_series[i:i+np.size(kernel)], kernel))
    
    return conv_series



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
    print(args.cutoff, type(args.cutoff[0]), args.cutoff[1])
    
    header = img.header
    TR = header.get_zooms()[3]
    
    
    voxel_size = header.get_zooms()[:3]
    f = open(args.output + ".txt", 'a')
    
    
    #print(data.shape, t, slicetimes.shape, TR, type(f))
    if args.slicetime!=None:
        slicetimes = np.array(np.loadtxt(args.slicetime[1]), dtype=np.float32)
        t = int(args.slicetime[0])
        data = slice_time_correction(data, t, slicetimes, TR*1000, f)
    if args.fwhm!=None:
        data = spatial_smoothing(data, int(args.fwhm), 5, voxel_size)
    if args.cutoff!=None:
        low = float(args.cutoff[0])
        high = float(args.cutoff[1])        
        data = temporal_filtering(data, TR, low, high)
    #print(fMRIData[5][5][5])
    saveNII = True
    outputImg = nb.Nifti1Image(data, np.eye(4), header=header)
    outputNIIFileName = args.output + '.nii.gz'
    nb.save(outputImg, outputNIIFileName)


                
