function [top_contacting, counts_for_top_contacting, number_of_cells_top_contacting, volume_for_top_contacting, ...
    centroid_for_top_contacting] = find_non_reported_neighboring_networks(relabeled_image,...
    chosen_region_label, how_many_to_report, X, Y, Z, Xorig, Yorig, Zorig)
% top_contacting -> unique IDs for networks -> entry 1:4 -> in order of how
%                   much they contact the region of interest

% counts_for_top_contacting -> number of voxels of that unique ID contacting
%                   the region of interest -> entries directly
%                   corresponding to entries in top_contacting


% This assumes all sims run to completion
%% read these in without taking as input -- make sure data are stored in appropriately name folder.

% check that appropriate folder exists
folder_with_big_sweep = '../input/archive_sweeps/3D/big_100x160x181_beta0_zpulse_divPlusMinus/';
assert(exist(folder_with_big_sweep) == 7);

unique_ID_filename = [folder_with_big_sweep, 'unique_net_id.npy'];
assert(exist(unique_ID_filename) == 2);

num_cell_filename = [folder_with_big_sweep, 'num_cells.npy'];
assert(exist(num_cell_filename) == 2);

isos_filename = [folder_with_big_sweep, 'isos.npy'];
assert(exist(num_cell_filename) == 2);

npy_state_tensor_unique_ID = readNPY(unique_ID_filename);
npy_state_tensor_num_cells = readNPY(num_cell_filename);
npy_state_tensor_isos = readNPY(isos_filename);

assert(isequal(size(npy_state_tensor_unique_ID), size(npy_state_tensor_num_cells)));
assert(isequal(size(npy_state_tensor_unique_ID), size(relabeled_image)));
assert(isequal(size(npy_state_tensor_isos), size(relabeled_image)));


%%

find_isos_nonzero = find(npy_state_tensor_isos(:)~=0);
isos_to_unique_IDs = unique(npy_state_tensor_unique_ID(find_isos_nonzero));
B_edited_image = zeros(size(relabeled_image));

this_region_counter = chosen_region_label;
stats1 = regionprops3(relabeled_image==this_region_counter,"VoxelIdxList");
voxID1 = stats1.VoxelIdxList{1,1};
[I1,I2,I3] = ind2sub(size(relabeled_image),voxID1);

for j = 1:length(voxID1)
    ind1 = I1(j);
    ind2 = I2(j);
    ind3 = I3(j);

    for ind1check = (ind1-1):(ind1+1)
        for ind2check = (ind2-1):(ind2+1)
            for ind3check = (ind3-1):(ind3+1)
                if (ind1check > 0 && ind1check <= size(relabeled_image,1) && ind2check > 0 && ind2check <= size(relabeled_image,2) && ind3check > 0 && ind3check <= size(relabeled_image,3))
                    B_edited_image(ind1check, ind2check, ind3check) = 1;
                end
            end
        end
    end

end

%
unique_list = unique(npy_state_tensor_unique_ID(find(B_edited_image(:)==1)));
unique_list = setdiff(unique_list, isos_to_unique_IDs);
count_vect = zeros(size(unique_list));
for i = 1:length(count_vect)
    count_vect(i) = length(find(npy_state_tensor_unique_ID(find(B_edited_image(:)==1)) == unique_list(i)));
end
[B,I] = sort(count_vect);

%
top_contacting = unique_list(I((end-(how_many_to_report-1)):end));
top_contacting = flipud(top_contacting);
counts_for_top_contacting = B((end-(how_many_to_report-1)):end);
counts_for_top_contacting = flipud(counts_for_top_contacting);
volume_for_top_contacting = zeros(size(counts_for_top_contacting));
centroid_for_top_contacting = zeros(length(counts_for_top_contacting), 3);

%
number_of_cells_top_contacting = zeros(size(top_contacting));
for i = 1:length(counts_for_top_contacting)
    unique_num_cells = unique(npy_state_tensor_num_cells(find(npy_state_tensor_unique_ID(:) == top_contacting(i))));
    assert(length(unique_num_cells) == 1);
    number_of_cells_top_contacting(i) = unique_num_cells;
end

% find biggest regions contacting -- calculate size and centroid.
for i = 1:length(counts_for_top_contacting)
    stats1 = regionprops3(npy_state_tensor_unique_ID==top_contacting(i),"VoxelIdxList","Volume");
    is_contacting_check = zeros(size(stats1,1),1);

    for j = 1:size(stats1,1)
        voxID1 = stats1.VoxelIdxList{j,1};
   
        intersect_check = intersect(voxID1, find(B_edited_image(:)==1));
        if (~isempty(intersect_check))
            is_contacting_check(j) = 1;
        end
    end
    find_is_contacting_check = find(is_contacting_check);
    [max_val,max_ind] = max(stats1.Volume(find_is_contacting_check));
    volume_for_top_contacting(i) = max_val;
    voxID = stats1.VoxelIdxList{find_is_contacting_check(max_ind),1};
    mean_val_pos = round([mean(Xorig(voxID)), mean(Yorig(voxID)), mean(Zorig(voxID))]);    
    centroid_for_top_contacting(i,:) = [X(mean_val_pos(2), mean_val_pos(1), mean_val_pos(3)), Y(mean_val_pos(2), mean_val_pos(1), mean_val_pos(3)), Z(mean_val_pos(2), mean_val_pos(1), mean_val_pos(3))];

end




end
