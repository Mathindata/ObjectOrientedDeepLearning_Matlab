load 'faces56x46.mat'
% "faces56x46.mat" is a 56x46x400 matrix
% There are 40 persons and for each one of them there are 10 pictures.
% Two sets of data are made. 
% SET1 for training and validation
% SET2 for testing
% There will be 4 output matrices.
% batchIdx<Ax3>, 
% testData<56x46x1xB>, testLabels<1xB>, 
% trainData<56x46x1x400-B>, trainLabels<1x400-B>
%
% Sets are ordered(not shuffled)

% Normalizing faces
faces = double(faces);
for i = 1:400
    faces(:,:,i) = faces(:,:,i) ./ norm(faces(:,:,i));
end
% Assigning a label to each picture
labels = zeros(1, size(faces, 3));
for i = 0:399
    labels(1, i+1) = idivide(int32(i), int32(10))+1;
end

% Setting the number of pictures from which SET1(for training) is composed,
% the remainder will be used for SET2(for testing)
set1Size = 100;
set2Size = size(faces, 3) - set1Size;


% SET1 : Training
trainData = faces(:, :, 1:set1Size);
trainData = reshape(trainData, 56, 46, 1, set1Size);
trainLabels = labels(1, 1:set1Size);

% SET2 : Testing
testData = faces(:, :, set1Size+1:set2Size + set1Size);
testData = reshape(testData, 56, 46, 1, set2Size);
testLabels = labels(1, set1Size+1:set2Size + set1Size);


% Producing a matrix to pair similar images.
% In each row: sample 1 index, sample 2 index, 0
for i = 1:set1Size
    for j = 1:10
        genIdx((i - 1) * 10 + j,:,:,:) = [i, idivide(int32(i-1), int32(10))*10 + j, 0];
    end
end

% Producing a matrix to pair dissimilar images.
% In each row: sample 1 index, sample 2 index, 1
for k = 1:set1Size
        st = idivide(int32(k-1), int32(10))*10 +1;
        en = st + 9;
        for l = st:en
            c=[1:st-1, en+1:set1Size]';
            imposIdx((set1Size-10)*(k-1)+1:(set1Size-10)*(k-1)+set1Size-10,:,:)=[k*ones(set1Size-10,1),c,ones(set1Size-10,1)];
        end
end

% A vector to shuffle dissimilar pairs indices
randIdx = randperm(size(imposIdx,1),set1Size*20);

% Catenating the similar and dissilar sample indices
% ratio of similar and dissimilar pairs is 1/1
batchIdx = vertcat(genIdx,imposIdx(randIdx,:,:));

% Shuffling the pairs in SET1
randS1 = randperm(size(batchIdx, 1), size(batchIdx, 1));
batchIdx = batchIdx(randS1,:,:);



clear c en faces genIdx i imposIdx j k l labels
clear randIdx randIdxTest randS1 set1Size set2Size st 
save 'atntpr56x46lecun.mat'