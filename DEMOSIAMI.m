fprintf('\nHere we train a Convolutional Multi-layered Neural Network\non the AT&T Faces dataset.\n\n')

% Output of atntpr56x46two.m  
% Loads: trainLabels, trainData, testLabels, testData, batchIdx
load 'atntpr56x46lecun.mat';

trainLabels = double(trainLabels);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining the architecture of the network
dataSize = [56,46,1];  % [nY x nX x nChannels]
hashLength = 50;
arch = {struct('type','input','dataSize',dataSize), ...
        struct('type','conv','filterSize',[7 7], 'nFM', 15), ...
        struct('type','subsample','stride',[2 2]), ...
        struct('type','conv','filterSize',[6 6], 'nFM', 45), ...
        struct('type','subsample','stride',[4 3]), ...
        struct('type','conv','filterSize',[1 1], 'nFM',250), ...
        struct('type','subsample','stride',1), ...
        struct('type','output', 'nOut', hashLength)};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


siamiNet = siamid(arch);
siamiNet.batchSize = 100;
siamiNet.nEpoch = 2;

% the last argument of train could be 'liti' or 'masih'
siamiNet = siamiNet.train(trainData, batchIdx, 'masih');









