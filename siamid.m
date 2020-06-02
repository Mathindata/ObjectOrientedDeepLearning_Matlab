classdef siamid
    properties
        arch=[]; % Architecture used for both networks
        batchSize;
        nEpoch; % number of epoch
        hashLength; % length of each network output layer
        costFun = 'xent'; % one of the costFuns used in mlcnnsiami
        net;
    end % properties
    
    methods
        function self = siamid(arch)
            self.arch = arch;
            self.hashLength = arch{end}.nOut;
        end % siamid
        
        function self = train(self, trainData, batchIdx, method)
        % Three arguments should be supplied.
        % trainData, batchIdx(in each row indices of a pair and silimarity value(0:similar,1:dissimilar))
        % method: currently: 'masih' and 'liti'
        
            net1 = mlcnnsiami(self.arch);
            net1.costFun = self.costFun;
            net2 = net1;
            
            % For method = 'liti'
            if strcmp(method, 'liti')
                s = self.calcSimIdxLiti(trainData, batchIdx);
            end
            
            
            tic
            for epoch = 1:self.nEpoch
            % Each time a batch with size "batchSize" is chosen from "batchIdx"
            fprintf('Epoch %d/%d\n',epoch,self.nEpoch);
                for iB = 1:self.batchSize:size(batchIdx)
                    % tBatch is indices of the current batch
                    tBatch = [iB:iB + self.batchSize - 1];
                    input1 = trainData(:,:,:,batchIdx(tBatch,1));
                    input2 = trainData(:,:,:,batchIdx(tBatch,2));
                    

                    % The two networks are fed forward with their corresponding inputs,
                    % then their outputs are saved in y1 and y2
                    net1 = net1.fProp(input1);
                    y1 = net1.netOut;
                    net2 = net2.fProp(input2);
                    y2 = net2.netOut;
                    
                    % dJ1 and dJ2 are derivatives of energy function with respect to
                    % output codes(y1 and y2)
                    switch method
                        case 'liti'
                            alpha = 1.25; % default value
                            dJ1 = self.calcDjLiti(y1, y2, tBatch, s, alpha);
                            dJ2 = self.calcDjLiti(y2, y1, tBatch, s, alpha);
                        case 'masih'
                            lambda = .001 * ones(self.hashLength, self.batchSize);
                            Target = batchIdx(tBatch,3)';
                            T = repmat(Target, [self.hashLength 1]);
                            dJ1 = self.calcDjMasih(y1, y2, T, lambda); 
                            dJ2 = self.calcDjMasih(y2, y1, T, lambda);           
                    end
                    
                    % Calculating the Energy function
                    % E = Target + (1-2Target)[(y1-y2)' * (y1-y2)] + lambda * (norm(y1) + norm(y2))

                    % Assigning the modified netError(dJ1 and dJ2) to networks and
                    % backpropagating to prepare properties(filter, dFilter, b, db)
                    net1.netError = dJ1;
                    net1 = net1.bProp();
                    net2.netError = dJ2;
                    net2 = net2.bProp();

                    % Summing the partial derivatives of the two networks and assigning
                    % them to the first one.
                    for i = 2:size(self.arch, 2) % number of layers
                        switch net1.layers{i}.type
                            case 'conv'
                                net1.layers{i}.filter = net1.layers{i}.filter + net2.layers{i}.filter;
                                net1.layers{i}.dFilter = net1.layers{i}.dFilter + net2.layers{i}.dFilter;
                                net1.layers{i}.b = net1.layers{i}.b + net2.layers{i}.b;
                                net1.layers{i}.db = net1.layers{i}.db + net2.layers{i}.db;
                            case 'subsample'
                                net1.layers{i}.b = net1.layers{i}.b + net2.layers{i}.b;
                                net1.layers{i}.db = net1.layers{i}.db + net2.layers{i}.db;
                            case 'output'
                                net1.layers{i}.dW = net1.layers{i}.dW + net2.layers{i}.dW;
                                net1.layers{i}.b = net1.layers{i}.b + net2.layers{i}.b;
                                net1.layers{i}.db = net1.layers{i}.db + net2.layers{i}.db;
                        end
                    end
                    
                    % Updating net1 and setting net2 parameters to the same values
                    net1 = net1.updateParams();
                    net2 = net1;
                end % batches
                toc
            end % epoch
            self.net = net1;
        end % train

        
        function dJ = calcDjMasih(self, y1, y2, T, lambda)  
        % Calculating "dj" which will be used as
            T = double(T);
            dJ1 = (y1 .* (1 - (2 .* T) + lambda)) - (1 - (2 .* T)) .* y2;
            dJ = dJ1;

        end % calcDjMasih

        function s = calcSimIdxLiti(self,trainData, batchIdx)
        % For liti method, "s" is a similarity value between each pair of samples
        % s = 0 if two pairs are dissimilar.
        % s = 1/(1+norm(x1-x2)) if similar. 
        
             x1 = trainData(:,:,:,batchIdx(:,1));
             x2 = trainData(:,:,:,batchIdx(:,2));

             x1 = reshape(x1, size(trainData,1)*size(trainData,2),size(batchIdx,1));
             x2 = reshape(x2, size(trainData,1)*size(trainData,2),size(batchIdx,1));

            distIns = diag(pdist2(x1', x2'));
            s = 1 ./ (1+distIns);
            s(batchIdx(:,3)==1)=0;
        end % calcSimIdxLiti
        
        function dJ = calcDjLiti(self, y1, y2, tBatch, s, alpha) 
        % Calculating "dJ" with Liti method
            
            % Initializing dJ1 with zeros
            dJ1 = zeros(self.hashLength, self.batchSize);
            
            %% First Condition: S >0
            % Find values in s which are bigger than zero and their indices
            [xx, ~, vv] = find(s(tBatch));
            %  Multiplying positive s values by their corresponding y values
            dJ1(:, xx) = bsxfun(@times,y1(:, xx) - y2(:,xx),vv');

            %% Second Condition: s == 0 and distOuts < alpha
            % Calculating distance of sample pairs
            distOuts = diag(pdist2(y1', y2'));

            % Some values from distOuts will be used for dividing,
            % So they shouldn't become zeros
            distOuts(distOuts == 0) = .0001;
            
            % Find indices for which distOuts<alpha and s==0
            dissimIndices = find(distOuts<alpha & s(tBatch)==0);
            tt = bsxfun(@rdivide, y1(:,dissimIndices)-y2(:,dissimIndices), distOuts(dissimIndices)');
            uu = bsxfun(@times, tt, distOuts(dissimIndices)'-alpha);

            dJ1(:,dissimIndices)= uu;
            %% Third Condition: s==0 and distOuts > alpha
            % For this condition, corresponding values in dJ should be zero
            % so since dJ is initialize as a zero matrix, there's nee need
            % to write any code.
            
            dJ = dJ1;
        end % calcDjLiti

    end % methods
end % classdef