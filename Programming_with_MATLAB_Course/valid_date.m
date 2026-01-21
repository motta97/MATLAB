function x=valid_date(yr,month,day)
%first we want to check that inputs are positive integers

if ( ~isscalar(yr) || yr < 1 || yr ~= fix(yr) ) || ( ~isscalar(month) || month < 1 || month ~= fix(month) ) ||( ~isscalar(day) || day < 1 || day ~= fix(day) )
    x=false;
    return
end

%first we want to check if it's a leap yr because it's going to depend on
%it
% a leep yr is anyyr that's divisible by 4 except for the yrs that are
% divisble by 100
%but yrs that are divisible by 400 is a leep yr, so how can we implement
%that in code?
if ( (mod(yr,4)==0 && mod(yr,100)~=0) || (mod(yr,400)==0))
    l=true;
else
    l=false;
end
if(l)
    %here the code goes for the case of leep yr
    %first we will check for the month and then the day
    %we will make sure that both of them are valid
    %the difference between a leep year and a normal one is that 
    %fabruary has 29 days instead of 28, but all the months are the same
    
        if(month==1)%january
            if(day<=31)
                x=true;
                return
            else
                x=false;
                return
            end
        
        elseif(month==2)%fab but for a leap year
            if(day<=29)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==3)%march
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==4)%abril
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==5)%may
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==6)%june 
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
       elseif(month==7)%july
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
         elseif(month==8)%august
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end 
        elseif(month==9)%sep
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
    elseif(month==10)%october
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==11)%november
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
         elseif(month==12)%december
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        else %greater than 12
        x=false;
        return
        end
else
        if(month==1)%january
            if(day<=31)
                x=true;
                return
            else
                x=false;
                return
            end
        
        elseif(month==2)%fab but for a leap year
            if(day<=28)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==3)%march
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==4)%abril
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==5)%may
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==6)%june 
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==7)%july
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
         elseif(month==8)%august
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end 
        elseif(month==9)%sep
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
    elseif(month==10)%october
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        elseif(month==11)%november
            if(day<=30)
                x=true;
                return
            else 
                x=false;
                return
            end
         elseif(month==12)%december
            if(day<=31)
                x=true;
                return
            else 
                x=false;
                return
            end
        else %greater than 12
            x=false;
            return
        end
end
