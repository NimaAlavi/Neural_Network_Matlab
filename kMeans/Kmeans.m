classdef Kmeans
   properties
      centroid
      dataIn
      label
      clusters
   end
   methods
      function obj = doClustring(obj)
         while(1)
            centroid_old = obj.centroid;
            obj = setClusterLable(obj);
            obj = setNewCentroid(obj);
            if obj.centroid == centroid_old
               break;
            end
         end
      end
      function obj = setCenterInitial(obj)
         num = size(obj.dataIn, 1);
         index = randperm(num, obj.clusters);
         obj.centroid = obj.dataIn(index, :);
      end
      
      function obj = setClusterLable(obj)
         num = size(obj.dataIn, 1);
         dis = zeros(num, obj.clusters);
         obj.label = zeros(num, 1);
         for i = 1: num
            for j = 1: obj.clusters
               dis(i, j) = norm(obj.dataIn(i,:) - obj.centroid(j, :), 2);
            end
         end
         for i = 1: num
            obj.label(i) = find(dis(i, :) == min(dis(i, :)));
         end
      end
      
      function obj = setNewCentroid(obj)
         obj.centroid = zeros(obj.clusters, size(obj.dataIn, 2));
         for i = 1: obj.clusters
            obj.centroid(i, :) = mean(obj.dataIn(obj.label==i, :));
         end
      end
   end
end