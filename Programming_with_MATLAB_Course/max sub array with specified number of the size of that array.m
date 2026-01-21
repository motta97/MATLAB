function [summa ,index]= max_sum(v,n)
s=size(v);
sum2=0;
if n<s(1,2)% checks whwether the n is greater than number of elements
    for i =1:s(1,2)
        if(i+n<=s(1,2)+1)
            for j=i:i+n-1
            sum2=sum2+v(1,j);
            end
            b(1,i)=sum2;
       
        end
        sum2=0;
         
        end

    [summa, index]=max(b);
elseif n==s(1,2)
    summa=sum(v);
    index=1;
else

    summa=0;
    index=-1;
end
