function x=saddle(m)
%the main function of this is to check all elements in the given matrix if
%they're saddle and ouput their postions in another matrix
%a saddle element is the element that's >= all of its elements in its row
%and <= all of the elements in its column
%we're going to compare each element with the others in its row and
%coulumn,we will assume that the element is saddle untill proved otherwise
[nr ,nc]=size(m);
c=0;
x=[];
flag=true;
flag2=true;
for n=1:nr
    for q=1:nc
        %now we know that we will access each element in the matrix, we
        %need to check here
        %first check on the rows
        for w=1:nc
            if m(n,q)<m(n,w)
                flag=false;
                break;
            end
            
        end
        %now check on the cols
        
            for w=1:nr

                if m(n,q)>m(w,q)
                flag2=false;
                break;
                end

            end
           if(flag &&flag2)
               %here we want to add the indices of that element to the
               %matrix
               %making it till here means that element is saddle
                c=c+1;
                x(c,1)=n;
                x(c,2)=q;
           end
    flag=true;
    flag2=true;
        end


    end
 
       
end
