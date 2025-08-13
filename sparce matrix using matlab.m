function x= sparse2matrix(c)
n=numel(c);
s=c{1};
def=c{2};
 if def==0
    x=zeros(s(1),s(2));
    else
    x=ones(s(1),s(2)).*def;
 end


for w=3:n
   
    v1=c{w};
    x(v1(1),v1(2))=v1(3);
end

   


    
