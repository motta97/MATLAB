function c=char_counter(text,ch)
c=0;
%the main function of this function is to count a given letter in a given
%text file
if exist(text,"file")~=2
    c=-1;
    return
else
    if ischar(ch)
        %here we type the counting code
        filecontent=fileread(text);
        charvector=char(filecontent);
        for n=1:length(charvector)
            if charvector(n)==ch 
                c=c+1;
       
            end
        end

        
    else
        c=-1;
        return
    end
end

