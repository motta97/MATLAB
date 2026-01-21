function sum=halfsum(a)
s=size(a);

    total=0;
    for n=1:s(1,1)
    
        for j=n:s(1,2)
            total =total+a(n,j);
        end
    end
    sum=total;



         
