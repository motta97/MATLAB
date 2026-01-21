function o=blur(i,w)
%we take the matrix of a given photo
%now we want to blur it by averaging each pixel with the elements besides
%it, you have to take care of pixels in the corners
%first we want to loop on each element on the matrix
i=double(i);
z=(2*w+1);
[nr, nc]=size(i);
o=zeros(nr,nc);
for n=1:nr
    for b=1:nc
        %now we know that we will access each element, it's time to average
        %each pixel
        sum=0;
        count=0;
        for x=n-w:n+w
            for y=b-w:b+w
                if x>=1 && (y)<=nc&& y>0 &&x<=(n+w)&&y<=(b+w)&&x<=nr
                    sum=sum+i(x,y);
                    count=count+1;
                end
            end
        end
        o(n,b)=sum/count;
            
            



       end

end
o=uint8(o);
end
