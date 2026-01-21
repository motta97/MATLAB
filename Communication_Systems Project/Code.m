%Analog Q1-4 - Shady Tarek - 2101513
pkg load signal;
fs=100;
df=0.01;
ts=1/fs;
N=ceil(fs/df);
t=-(N*ts)/2:ts:(N*ts)/2-ts;

% Define the function x(t)
x = zeros(size(t));
for i = 1:length(t)
    if abs(t(i)) <= 4
        x(i) = -abs(t(i)) + 5;
    end
end
subplot(2,1,1);
figure(1);
plot(t, x);
xlabel('time'); ylabel('x(t)');
title('Time Domain');
grid on;
xlim([-5,5]);

% Define the frequency domain
if (rem(N,2)==0)
    f = - (0.5*fs) : df : (0.5*fs-df) ;
else
    f = - (0.5*fs-0.5*df) : df : (0.5*fs-0.5*df) ;
end

% Define the function x(f)analyticaly
Y= 8*sinc(8*f)+16*(sinc(4*f)).^2;

% Fourier transform of x(t)
X= fftshift(fft (x)) *ts;
figure(1)
subplot(2,1,2);
plot(f,abs(X));
hold on;
plot(f,abs(Y));
xlabel('freq'); ylabel('X(f)');
title('Fourier Transform ');
grid on;
xlim([-5,5]);
legend('Fourier(octave)', 'Fourier(analytical)');

% Estimate the bandwidth
power_total=sum(abs(X).^2)*df;
index= find(f==0);
power_acc=0; %accimulator
for c_index = index: length(f)
power_acc=df*abs (X(c_index)).^2+power_acc;
if(power_acc>=0.95*0.5*power_total)
BW= f(c_index);
break
end
end


% Analog Q5-7 - Mahmoud Nabil Abdelrhman - 2201051
% Q.5  we need first to implemnt the LPF with 1hz BW
H = double(abs(f) < 1);
figure(2)
plot(f,H) ;
xlabel('Frequncy') ;
ylabel('H(F)');
title('LPF');
grid on;
%then multiply the LPF with the signal in freqyncy domain
J = X .* H ;
j = real(ifft(fftshift(J))*N);
figure(3)
plot(t,j);
xlabel('time');
ylabel('j(t)');
title('response of the signal on LPF in time domain');
grid on ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Q.6 if the LPF reduced to 0.3 hz
H = double(abs(f) < 0.3);
figure(4)
plot(f,H) ;
xlabel('Frequncy') ;
ylabel('H(F)');
title('LPF');
grid on;
J = X .* H ;
j = real(ifft(fftshift(J))*N);
figure(5)
plot(t,j);
xlabel('time');
ylabel('j(t)');
title('response of the signal on LPF in time domain');
grid on ;



% Q.7
fm7 =0.5;
fs7=100;
ts7=1/fs7;
t7=0:ts7: 4;
N7=length(t7);
df7=fs7/N7;




%define the freqncy
 if(rem(N7,2)==0)
f = (-N7/2:N7/2-1)*(fs7/N7);
 else
f = - (0.5*fs7-0.5*df7) : df7 : (0.5*fs7-0.5*df7);
 end
f= f(1:N7);
%Define the function x (t7)
 x= cos(2*pi*fm7*t7);
 figure(6)
 plot (t7, x);
 xlabel('time'); ylabel('x(t7)');
 title('Time Domain');
 grid on;



%Define the function x (f)analyticaly
Y = 2 * (sinc(4*(f - fm7)) + sinc(4*(f + fm7)));
%Fourier transform of x(t7)
X= fftshift(fft (x)) *ts7;
figure(7)
plot (f, abs (X));
hold on;
plot(f,abs(Y));
xlabel('freq'); ylabel('X(f)');
title('Fourier Transform ');
grid on;
legend('Fourier (octave)', 'Fourier (analytical)');

 %Estimate the bandwidth B

 power_total=sum(abs (X).^2) *df7;
 index = find(f==0);
 power_acc=0;%accimulator
for c_index = index: length (f)
 power_acc=df7*abs (X(c_index)).^2+power_acc;
 if (power_acc>=0.95*0.5*power_total)
 BW= f(c_index);

 break
 end
end







% Analog Q8-12 - Ganna Khaled Abd ElNasser  - 2200089
% Parameters
fs = 1000;
T = 4;
t = 0:1/fs:T-1/fs;
N = length(t);

x = sin(2*pi*0.8*t); % (from step 5)
m = sin(2*pi*1.5*t);

fc1 = 20; % DSB-SC
fc2 = fc1 + 2 + 2; % SSB (2 Hz bandwidth each + 2 Hz guard band = 24 Hz)

% DSB-SC modulation of x(t)
c1 = cos(2*pi*fc1*t);
s1 = x .* c1;

% SSB (USB) modulation of m(t)
c2 = cos(2*pi*fc2*t);
m_hilbert = imag(hilbert(m)); % Hilbert transform
s2 = m .* c2 - m_hilbert .* sin(2*pi*fc2*t);


%Analog Q8-12 - Abdallah Hossam Ragab - 2100844
% Q11 - Plot s(t) = s1(t) + s2(t) and S(f)

pkg load signal   % Load signal package for hilbert()

% Parameters
fs = 1000;            % Sampling frequency
T = 4;                % Time duration in seconds
t = 0:1/fs:T-1/fs;    % Time vector
N = length(t);        % Number of samples

% Messages
x = sin(2*pi*0.8*t);     % x(t)
m = sin(2*pi*1.5*t);     % m(t)

% Carrier frequencies
fc1 = 20;                 % DSB-SC carrier
fc2 = fc1 + 2 + 2;        % SSB carrier with 2 Hz guard = 24 Hz

% Modulation
c1 = cos(2*pi*fc1*t);
s1 = x .* c1;

c2 = cos(2*pi*fc2*t);
m_hilbert = imag(hilbert(m));
s2 = m .* c2 - m_hilbert .* sin(2*pi*fc2*t);

% Combined FDM signal
s = s1 + s2;

% --- Plot s(t) ---
figure(8);
subplot(2,1,1);
plot(t, s);
title('Combined Signal s(t) = s_1(t) + s_2(t)');
xlabel('Time (s)');
ylabel('Amplitude');

% --- Frequency Spectrum S(f) ---
% FFT and frequency axis
S = abs(fftshift(fft(s)));
f = (-N/2:N/2-1)*(fs/N);
subplot(2,1,2);
plot(f, S);
title('Magnitude Spectrum |S(f)|');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 50]);   % Focus on relevant spectrum range

% Load required package for filter and Hilbert transform
pkg load signal

% Parameters
fs = 1000;              % Sampling frequency
T = 4;                  % Duration in seconds
t = 0:1/fs:T-1/fs;      % Time vector
N = length(t);          % Number of samples

% Messages
x = sin(2*pi*0.8*t);     % Message 1
m = sin(2*pi*1.5*t);     % Message 2

% Carrier frequencies
fc1 = 20;                % Carrier for DSB-SC (x)
fc2 = fc1 + 2 + 2;       % Carrier for SSB (m) + 2 Hz guard band => 24 Hz

% Modulation - DSB-SC for x(t)
c1 = cos(2*pi*fc1*t);
s1 = x .* c1;

% Modulation - SSB (USB) for m(t)
c2 = cos(2*pi*fc2*t);
m_hilbert = imag(hilbert(m));
s2 = m .* c2 - m_hilbert .* sin(2*pi*fc2*t);

% Total FDM signal
s = s1 + s2;

% -------------------------------
% Coherent Demodulation
% -------------------------------

% DSB-SC Demodulation of x(t)
x_demod = 2 * s1 .* c1;

% SSB (USB) Demodulation of m(t)
m_demod = 2 * s2 .* c2;

% Low-pass filter design (Butterworth, order 6, cutoff 2 Hz)
[b, a] = butter(6, 2/(fs/2));

% Filtering to recover signals
x_rec = filter(b, a, x_demod);
m_rec = filter(b, a, m_demod);

% -------------------------------
% Plotting Results
% -------------------------------

figure(9);

subplot(2,1,1);
plot(t, x, 'b', t, x_rec, 'r--');
title('Original x(t) vs Recovered x(t) (DSB-SC)');
legend('Original x(t)', 'Recovered x(t)');
xlabel('Time (s)');
ylabel('Amplitude');
subplot(2,1,2);
plot(t, m, 'b', t, m_rec, 'r--');
title('Original m(t) vs Recovered m(t) (SSB - USB)');
legend('Original m(t)', 'Recovered m(t)');
xlabel('Time (s)');
ylabel('Amplitude');



%Digital part 1 - Nour Ayman Mahmoud 2200105
% Parameters
num_bits = 64;
bit_duration = 1;
sampling_rate = 100;
fs = sampling_rate / bit_duration;

% Generate random bit stream
bits = randi([0, 1], 1, num_bits);

% Time vector for one bit
t_bit = linspace(0, bit_duration, sampling_rate);


% Polar NRZ Encoding

polar_nrz = [];
for bit = bits
    if bit == 1
        pulse = ones(1, length(t_bit));
    else
        pulse = -ones(1, length(t_bit));
    end
    polar_nrz = [polar_nrz, pulse];
end


% Manchester Encoding

manchester = [];
for bit = bits
    if bit == 1
        % First half high, second half low
        pulse = [ones(1, length(t_bit)/2), -ones(1, length(t_bit)/2)];
    else
        % First half low, second half high
        pulse = [-ones(1, length(t_bit)/2), ones(1, length(t_bit)/2)];
    end
    manchester = [manchester, pulse];
end

% Time vector for entire signal
t_total = linspace(0, num_bits * bit_duration, length(polar_nrz));


% Time Domain Plots

figure(10);
subplot(2,1,1);
plot(t_total, polar_nrz, 'LineWidth', 1.5);
title('Polar NRZ Encoding - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1.5 1.5]);
grid on;

subplot(2,1,2);
plot(t_total, manchester, 'LineWidth', 1.5);
title('Manchester Encoding - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1.5 1.5]);
grid on;


% Frequency Domain Analysis

% Compute FFT
n = 2^nextpow2(length(polar_nrz));
f = (-n/2:n/2-1) * (fs/n);

% Polar NRZ spectrum
polar_fft = abs(fftshift(fft(polar_nrz, n)));
polar_fft = polar_fft / max(polar_fft); % Normalize

% Manchester spectrum
manchester_fft = abs(fftshift(fft(manchester, n)));
manchester_fft = manchester_fft / max(manchester_fft); % Normalize

% Frequency domain plots
figure(11);

% Polar NRZ spectrum plot
subplot(2,1,1);
plot(f, polar_fft, 'LineWidth', 1.5);
title('Polar NRZ Encoding - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Normalized Magnitude');
xlim([-10 10]);
grid on;

% Manchester spectrum plot
subplot(2,1,2);
plot(f, manchester_fft, 'LineWidth', 1.5);
title('Manchester Encoding - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Normalized Magnitude');
xlim([-10 10]);
grid on;


%Digital part 2 - Mostafa Ahmed Abd ELSattar - 2201092

%defining variables for digital analysis
%defining variables for digital analysis
tb=0.02;
rb=1/tb;
numOfBits=64;
T = numOfBits*tb;
N_digital = ceil( T/tb);
t_digital=repelem(0:tb:N_digital*tb, 2);
t_digital = t_digital(2:end-1);
df_digital = rb/N_digital;
f_digital = (-floor(N_digital/2) : ceil(N_digital/2)-1) * df_digital;




%defining the carrier signal
fc=10/tb;
T_analog=0.002;
ts=0.01/fc;
N_analog=ceil(T_analog/ts);
t=0:ts:(N_analog-1)*ts;
b_tx = sqrt(2/tb)*cos(2*pi*fc*t);

%generating random bits
random_bits= randi([0 1] ,1,numOfBits);

%defining frequency
fs=1/ts;
df = fs/N_analog;
if(rem(N_analog,2)==0)
f= -fs/2:df:fs/2-df;
else
f= -[fs/2 -df/2]: df: [fs/2 -df/2];
end

%%%%%%%%%TX%%%%%%%%%%
%Polar NRZ encoder
max_length= length(random_bits);
for c = 1:max_length
  if random_bits(c) ==0
    random_bits(c)=-1;
  endif
endfor

parent = cell(1,max_length);
%the modulated signal
for c = 1: max_length
parent{c} = b_tx.*random_bits(c);
endfor

%Plotting in time
tg= t;
figure(12)
for i=1:length(parent)
  plot(tg,parent{i})
  tg+=tb/10;
  hold on
end
xlabel("Time");
ylabel("Amplitude");
title("BPSK in time domain");




%getting frequency representation
parentF= cell(1,max_length);
for c = 1: max_length
parentF{c} = fftshift(fft(parent{c}))*ts;
endfor

%plotting in frequency
figure(13)
for i=1:length(parent)
  stem(f,parentF{i})
  hold on
end
xlabel("Frequency");
ylabel("Amplitude");
title("BPSK in Frequency domain");

%%%%%%%%%%%%%RX%%%%%%%%%%%%

%multiplying the signal by the basis function
phi=pi/2;
b_rx=sqrt(2/tb)*cos(2*pi*fc*t + phi);
for c = 1: max_length
parentR{c} = parent{c}.*b_rx;
endfor



%doing integral
parentIntegaral = cell(1,max_length);
for c=1:max_length
  parentIntegaral{c}=cumtrapz(tb,parentR{c});
end

%decision making with threshold = 0;
%the bit that was originally zero has only negative values and one has only positive
%so we check on the second element only
parentFinal=zeros(1,max_length);
for c = 1:max_length
   if(parentIntegaral{c}(2)>0)
   parentFinal(c)=1;
 else parentFinal(c)=0;
 end
end



%finindng frequency representation
parentFinalFrequency=fftshift(fft(parentFinal))*ts;

%plotting in time
figure(14)
bit_plot = repelem(parentFinal, 2); % Two points per bit
t_digital = linspace(0, numOfBits * tb, length(bit_plot));%modifiying t_digital for correct plot

stairs(t_digital, bit_plot, 'LineWidth', 2);
ylim([-0.5 1.5]); grid on;
xlabel('Time'); ylabel('Bit value');
title('Digital Signal vs Time');


%plotting in frequency
figure(15)
plot(f_digital,parentFinalFrequency)
xlabel("Frequency");
ylabel("Amplitude");
title("Spectrum at the Output of the reciever");
