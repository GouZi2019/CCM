function ClusterIndex = get_collected_points(X, V, k_ratio, z_ratio, t_ratio, elastic, jac, debug)

[X,index,~]  = unique(X, 'rows', 'stable');
V = V(index, :);

dim = size(X,2);

if debug
    save('temp/for_debug_input.mat')
end

%% Collectiveness parameter
if k_ratio>1
    para.K = k_ratio;
else
    para.K = max(ceil(size(X, 1)*k_ratio), 2);
end
para.z = z_ratio/para.K ;
para.upperBound = para.K*para.z/(1-para.K*para.z);
para.threshold = t_ratio*para.z/(1-para.K*para.z);
para.e = double(elastic);
para.j = double(jac);

%% step 1: compute the weighted adjacency matrix using KNN
weightedAdjacencyMatrix = computeAdj(X, V, para.K, para.e, para.j);

%% step 2: integrating all the paths with regularization
I_matrix = eye(size(weightedAdjacencyMatrix,1));
Z = inv(I_matrix-para.z*weightedAdjacencyMatrix) - I_matrix;

linkGraph = Z > para.threshold;
[ ClusterIndex ] = linkGraph2cluster( linkGraph, dim );

if debug
    save('temp/for_debug_output.mat')
end

end

function [ clusterIndex ] = linkGraph2cluster( linkGraph, dim )
[row,col] = find(linkGraph==1);
pairData = [row';col'];
clusterIndex = pairClustering(pairData,size(linkGraph,1),dim);
end


function clusterIndex = pairClustering(pairwiseData,totalNum,dim)

clusterNum = 0;
clusterIndex = zeros(1,totalNum);

for i = 1:size(pairwiseData,2)
    curPair = pairwiseData(:,i);
    curPairAlabel = clusterIndex(1,curPair(1));
    curPairBlabel = clusterIndex(1,curPair(2));
    
    if curPairAlabel == 0 && curPairBlabel == 0
        clusterNum = clusterNum+1;
        curPairLabel = clusterNum;
        clusterIndex(1,curPair(1)) = curPairLabel;
        clusterIndex(1,curPair(2)) = curPairLabel;
    elseif curPairAlabel~=0 && curPairBlabel==0
        clusterIndex(1,curPair(2)) = curPairAlabel;
    elseif curPairBlabel~=0 && curPairAlabel==0
        clusterIndex(1,curPair(1)) = curPairBlabel;
    else
        combineLabel = min(curPairAlabel,curPairBlabel);
        clusterIndex(1,clusterIndex==curPairAlabel) = combineLabel;
        clusterIndex(1,clusterIndex==curPairBlabel) = combineLabel;
    end
    

end

newClusterNum = 0;
for i = 1:max(clusterNum)
    curClusterIndex = find(clusterIndex==i);
    if length(curClusterIndex)< 10
        clusterIndex(curClusterIndex) = 0;   
    else
        newClusterNum = newClusterNum+1;
        clusterIndex(curClusterIndex) = newClusterNum;
    end
end
    
end

function weightedAdjacencyMatrix = computeAdj(curX, curV, K, elastic, jac)
point_num = size(curX,1);
point_dim = size(curX,2);

pfs = curX';
pms = curX'+curV';
movement = curV';

distanceMatrixBefore = slmetric_pw(pfs,pfs,'eucdist');
distanceMatrixAfter = slmetric_pw(pms,pms,'eucdist');
%% K-nearest neighbor adjacency matrix
neighborMatrix = zeros(point_num);
neighborLists = {};
for i=1:size(curX,1)
    [~,neighborIndexBefore] = sort(distanceMatrixBefore(i,:),'ascend');
    [~,neighborIndexAfter] = sort(distanceMatrixAfter(i,:),'ascend');
    
    % 两个index求交集
    % two index are sorted, so the first one is the closest point
    % 包含自身，得到交叠的所有点
    % include itself, get all the overlapping points
    neighborIndex = intersect(neighborIndexBefore(1:K+1), neighborIndexAfter(1:K+1), 'stable');  % 这有个大坑，它返回会默认排序，所以要加stable
    % 不一定有k个了，从第二个开始到最后，赋值为1
    % not necessarily k, from the second to the last, assign 1
    neighborMatrix(i,neighborIndex(2:end)) = 1;
    neighborLists{i} = neighborIndex;
end

if jac == -1
    jacCorrelationVector = ones(point_num, 1);
else
    jacCorrelationVector = zeros(point_num, 1);
    roleVector = zeros(point_num, 1);
    while 1
        % 先拿到本轮迭代初始未定义索引，如果本轮迭代结束没有变化，未定义就都被视为Evil
        % First get the initial undefined index of this round of iteration. If there is no change at the end of this round, the undefined ones are all considered Evil
        global_undefined_index = find(roleVector==0);
        global_good_index = find(roleVector==1);
        global_evil_index = find(roleVector==-1);

        for iter = 1:length(global_undefined_index)
            i = global_undefined_index(iter);
            neighbor_without_evil = setdiff(neighborLists{i}, global_evil_index, 'stable');
            neighbor_with_good = intersect(neighborLists{i}, global_good_index, 'stable');
            if length(neighbor_without_evil) < point_dim+2 % 插值用的控制点需要大于dim 1，由于这个neighbor中包含自己，所以大2  # The number of control points used for interpolation needs to be greater than dim 1. Since this neighbor contains itself, it is large 2
                % 连足够多的好亲戚都没有，直接标狼
                % There are not enough good relatives, directly mark the evil
                roleVector(i) = -1;
            elseif length(neighbor_with_good) >= point_dim+1 % 只包含good数量够插值，那就只用good; 注意，这里是交集，所以只包含好点，肯定不包含自己，所以只大1就行 # Only the number of good points is enough for interpolation, so only good points are used; Note that this is an intersection, so it only contains good points, and it must not contain itself, so it is only large 1
                [role, jacCorrelation] = GetJacCorrelation(pfs(:,[i;neighbor_with_good]), pms(:,[i;neighbor_with_good]), jac); % 根据上条，这里还得把自己加上送进去 # According to the previous article, I have to add myself to send it
                roleVector(i) = role;
                jacCorrelationVector(i) = jacCorrelation;
            else
                [role, jacCorrelation] = GetJacCorrelation(pfs(:,neighbor_without_evil), pms(:,neighbor_without_evil), jac); % 根据上条，这里还得把自己加上送进去 # According to the previous article, I have to add myself to send it
                roleVector(i) = role;
                jacCorrelationVector(i) = jacCorrelation;            
            end
        end

        result_undefined_index = find(roleVector==0);
        % 如果undefined完全没有改变，那么就把所有undefined都标狼，然后结束；否则进入下一轮。
        % If undefined is completely unchanged, then mark all undefined evils, and then end; otherwise enter the next round.
        if isempty(setxor(result_undefined_index, global_undefined_index))
            roleVector(result_undefined_index) = -1;
            jacCorrelationVector(result_undefined_index) = 0;
            break;
        end
    end
end

jacCorrelationMatrix = jacCorrelationVector*jacCorrelationVector';


if elastic == -1
    distanceCorrelationMatrix = ones(size(distanceMatrixAfter));
else
    distanceCorrelationMatrix = exp(-(distanceMatrixAfter-distanceMatrixBefore).^2./(elastic*distanceMatrixBefore.^2));
    distanceCorrelationMatrix(isnan(distanceCorrelationMatrix))=1;
end

correlationMatrix = slmetric_pw(movement,movement,'nrmcorr');
weightedAdjacencyMatrix = (correlationMatrix.*neighborMatrix.*distanceCorrelationMatrix.*jacCorrelationMatrix);%weigted adjacency matrix
end


function [role, jac_correlation] = GetJacCorrelation(pfs, pms, a)
  
% 第一个点是要检测的点
% The first point is the point to be detected
check_point = pfs(:,1);

try
    jac_det_before = jac_of_tps(pfs(:,2:end), pms(:,2:end),check_point);
    jac_det_after = jac_of_tps(pfs, pms,check_point);
catch
    role = -1; jac_correlation = 0;
    return
end

if jac_det_before <= 0
    role = 0; jac_correlation = 0;
    return;
end

if jac_det_after < 0
    role = -1; jac_correlation = 0;
    return;
end

jac_ratio = jac_det_after/jac_det_before;
role = 1;
if jac_ratio < 1
    jac_correlation = -1*(1-jac_ratio)^a+1;
else
    jac_correlation = exp(-(jac_ratio-1)^2/(a^2));
end

if isnan(jac_det_before*jac_det_after*jac_correlation)
   disp(1)
end

end