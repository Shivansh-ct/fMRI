import nibabel as nb
import numpy as np
import argparse
from scipy.fftpack import fft, ifft



def temporal_filtering_operation(data, TR, low, high):
    temp_filtered = np.zeros(data.shape)
    if low>high:
        temp = high
        high = low
        low = temp
    high_pass = 1/low
    low_pass = 1/high
    freq = np.fft.fftfreq(data.shape[3], d=1.0/(2.0*TR))
    #freq = np.concatenate((np.linspace(0,1.0/(2.0*TR),data.shape[3]//2),np.linspace(-1.0/(2.0*TR),0,data.shape[3]//2)))
    for dim_x in range(data.shape[0]):
        for dim_y in range(data.shape[1]):
            for dim_z in range(data.shape[2]):
                single_voxel_timeseries = data[dim_x][dim_y][dim_z][:]
                single_voxel_fft = fft(single_voxel_timeseries)
                single_voxel_fft[np.abs(freq)>high_pass]=0
                single_voxel_fft[np.abs(freq)<low_pass]=0
                single_voxel_ifft = ifft(single_voxel_fft)
                temp_filtered[dim_x][dim_y][dim_z][:] = single_voxel_ifft
    return temp_filtered


#t is the Target Time
def slice_time_correction_operation(data, t, slicetimes, TR, file_descriptor):
    c_img = np.zeros(data.shape)
    if t>TR or t<0:
        file_descriptor.write("SLICE TIME CORRECTION FAILURE")
        return data
        
    else:
        time = np.arange(0,data.shape[3],1)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    
                        if slicetimes[z]>TR or slicetimes[z]<0:
                            file_descriptor.write("SLICE TIME CORRECTION FAILURE")
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
                        c_img[x][y][z][0] = data[x][y][z][0]
    #c_img[0:][0:][0:][data.shape[3]-1] = data[0:][:][:][data.shape[3]-1]
    file_descriptor.write("SLICE TIME CORRECTION SUCCESS\n")
    #c_img[0][:][:][:] = data[0][:][:][:]
    #print(c_img.shape, data.shape, data.shape[3]-1)
    #a = int(data.shape[3]-1)
    #c_img[a][:][:][:] = data[a][:][:][:]
    return c_img
    
    
    
    

    
def spatial_smoothing_operation(data, fwhm, voxel_size):
    shape = data.shape
    smooth = np.zeros(shape)
    for time in range(0,data.shape[3]):
       #Applying the operation along x direction
       kernel_x = make_kernel_operation(fwhm, data.shape[0], voxel_size[0])
       for dim_y in range(0,data.shape[1]):
           for dim_z in range(0,data.shape[2]):
               smooth[:,dim_y,dim_z,time] = convolving_kernel_operation(data[:,dim_y,dim_z,time], kernel_x)
       #Applying the operation along y direction
       kernel_y = make_kernel_operation(fwhm, data.shape[1], voxel_size[1])
       for dim_x in range(0,data.shape[0]):
           for dim_z in range(0,data.shape[2]):
               smooth[dim_x,:,dim_z,time] = convolving_kernel_operation(smooth[dim_x,:,dim_z,time], kernel_y)
       #Applying the operation along z direction
       kernel_z = make_kernel_operation(fwhm, data.shape[2], voxel_size[2])
       for dim_y in range(0,data.shape[1]):
           for dim_x in range(0,data.shape[0]):
               smooth[dim_x,dim_y,:,time] = convolving_kernel_operation(smooth[dim_x,dim_y,:,time], kernel_z)
    #smooth represents the spatially smoothened image
    return smooth
    

#Preparing a Gaussian Kernel
def make_kernel_operation(fwhm, kernel_size, voxel_size):
    sigma = fwhm/(np.sqrt(8*np.log(2))*voxel_size)
    w = np.exp(-(np.arange(-kernel_size, kernel_size+1))**2/(2*(sigma**2)))
    kernel = w/sum(w)
    return kernel





def convolving_kernel_operation(series, kernel):
    padded_series = np.concatenate((np.zeros(np.size(kernel)//2), series, np.zeros(np.size(kernel)//2)))
    conv_series = np.zeros(series.shape)
    for i in range(series.shape[0]):
        conv_series[i] = np.sum(np.multiply(padded_series[i:i+np.size(kernel)], kernel))
    return conv_series



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
    parser.add_argument("-tc", "--slicetime", nargs=2, help="target time and slice time acquisition file")
    parser.add_argument("-tf", "--cutoff", nargs=2, help="temporal filtering high and low pass")
    parser.add_argument("-sm", "--fwhm", help="spatial smoothing fwhm")
    #parser.add_argument("-ts", "--timeseries", nargs=3, help="time series")
    args = parser.parse_args()
   # print(args.input, args.output, args.cutoff, args.slicetime, args.fwhm)
    
    
    fmri_image = nb.load(args.input)
    data = fmri_image.get_data()
    #print(args.cutoff, type(args.cutoff[0]), args.cutoff[1])
    
    header = fmri_image.header
    TR = header.get_zooms()[3]
    
    
    voxel_size_list = header.get_zooms()[:3]
    #Opening the File in Write Mode
    file_descriptor = open(args.output + ".txt", 'w')
    
    
    #print(data.shape, t, slicetimes.shape, TR, type(f))
    if args.slicetime!=None:
        slicetimes = np.loadtxt(args.slicetime[1])
        t = float(args.slicetime[0])
        data = slice_time_correction_operation(data, t, slicetimes, TR*1000, file_descriptor)
        
    if args.cutoff!=None:
        low = float(args.cutoff[0])
        high = float(args.cutoff[1])        
        data = temporal_filtering_operation(data, TR, low, high)
        
    if args.fwhm!=None:
        data = spatial_smoothing_operation(data, int(args.fwhm),voxel_size_list)
    #print(fMRIData[5][5][5])
    
    data = np.array(data, dtype=header.get_data_dtype())
    Output_Array = nb.Nifti1Image(data, np.eye(4), header=header)
    
    Output_NIFTI_FileName = args.output + '.nii.gz'
    
    nb.save(Output_Array, Output_NIFTI_FileName)
    #Closing the File before the program terminates
    file_descriptor.close()

                
