%reading the audio file



[signal_ft,fs] = audioread("ST.ogg");



%plotting in time domain


x = length(signal_ft);

t = (0:x-1)/fs;
figure('Name', 'Time Domain', 'NumberTitle', 'off');

plot(t,signal_ft)
xlabel('Time')
ylabel('signal')
title("Plot in time domain")


%calculating energy in time domain


Energy_in_time = sum(signal_ft.^2);



%Finding the frequency domain representaion and plotting it

figure('Name', 'Frequency Domain', 'NumberTitle', 'off');
freq_domain = fft(signal_ft);
l=length(freq_domain);
f=(0:l-1)*(fs/l);
plot(f,abs(freq_domain))
xlabel('frequancy')
ylabel('Signal')
title('Plot in frequency domain')


%calculating the signal energy in frequency domain

energy_frequency_domain = sum(abs(freq_domain).^2) / l;

%applying the filter
fc=800; %cut off frequency
[b,a] = butter(10,fc/(fs/2),'low'); %low pass filter of 10th order

filtered_signal=fft(filter(b,a,signal_ft));

f=(0:N-1)*(fs/N);
figure('Name', 'Signal in frequency after applying a filter', 'NumberTitle', 'off');
plot(f,abs(filtered_signal))
xlabel("frequency");
ylabel("Signal");
title("Filtered Signal in Fr")



%converting it to time domain
final_audio=ifft(filtered_signal);
figure('Name', 'Signal in time after applying a filter', 'NumberTitle', 'off');
plot(t,final_audio)
xlabel('Time')
ylabel('Signal')
title("Plot in time domain")

%writing it into PC
audiowrite("final audio.ogg",final_audio,fs)