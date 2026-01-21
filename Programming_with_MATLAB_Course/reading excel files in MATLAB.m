function d=get_distance(c1,c2)
n1=readcell('Distances.xlsx');
rowIdx = find(strcmp(n1(:, 1), c1)); % Assuming first column has row labels
colIdx = find(strcmp(n1(1, :), c2)); % Assuming first row has column labels
if isempty(rowIdx)||isempty(colIdx)
    d=-1;
else
    d=n1(rowIdx,colIdx);
    d=d{1};
end
