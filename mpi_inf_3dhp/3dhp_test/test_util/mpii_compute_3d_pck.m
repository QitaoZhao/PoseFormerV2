function [pck_table, auc_table] = mpii_compute_3d_pck(error_data, joint_groups, output_base_path)

%Input
%error_data is a struct array of type mpii_3d_error
%The struct zcarries information about the name of the method as well as an
%nj x 1 x nf matrix with the joint errors.
%joint_groups is an ng x 2 cell, where ng is the number of groups. It
%carries the name of the group as well as the indices of the joints that
%belong to the group.

%If the error_data array has multiple inputs, there are additional
%comparative AUC plots output per joint in addition to the individual ones.
ng = size(joint_groups,1);


pck_curve_array = cell(length(error_data), ng+1); %Contains the PCK results per joint group, per error_data cell
pck_array = cell(length(error_data), ng+1); %Contains the AUC results per joint group
auc_array = cell(length(error_data), ng+1); %Contains the AUC results per joint group
%thresh = 0:5:200;
thresh = 0:5:150;
pck_thresh = 150;


for i = 1:length(error_data)
    joint_count = 0;
    nf = size(error_data(i).error,3);
     for j = 1:ng
         for ti =1:length(thresh)
             t = thresh(ti);
             pck_curve_array{i,j} = [pck_curve_array{i,j}, sum(sum(error_data(i).error(joint_groups{j,2},1,:) < t, 3),1) / (length(joint_groups{j,2}) *nf)];
         end
         
         joint_count = joint_count + length(joint_groups{j,2});
         if(isempty(pck_curve_array{i,ng+1}))
             pck_curve_array{i,ng+1} = pck_curve_array{i,j} * length(joint_groups{j,2});
         else
             pck_curve_array{i,ng+1} = pck_curve_array{i,ng+1} + pck_curve_array{i,j} * length(joint_groups{j,2});
         end
         auc_array{i,j} = 100* sum(pck_curve_array{i,j}(:))/ length(thresh);
         pck_array{i,j} = 100* sum(sum(error_data(i).error(joint_groups{j,2},1,:) < pck_thresh, 3),1) / (length(joint_groups{j,2}) *nf);
         if(isempty(pck_array{i,ng+1}))
             pck_array{i,ng+1} = pck_array{i,j} * length(joint_groups{j,2});
         else
             pck_array{i,ng+1} = pck_array{i,ng+1} + pck_array{i,j} * length(joint_groups{j,2});
         end
     end
     pck_array{i,ng+1} = pck_array{i,ng+1} / joint_count;
     pck_curve_array{i,ng+1} = pck_curve_array{i,ng+1} / joint_count;
     auc_array{i,ng+1} = 100* sum(pck_curve_array{i,ng+1}(:))/ length(thresh);
end
         
pck_table = cell(length(error_data)+1, ng+2);
pck_table{1,ng+2} = 'Total';
for i = 1:length(error_data)
    pck_table{1+i,1} = error_data(i).method;
end
for i = 1:ng
    pck_table{1,i+1} = joint_groups{i,1};
end
auc_table = pck_table;
auc_table(2:end,2:end) = auc_array;
pck_table(2:end,2:end) = pck_array;


if(~isempty(output_base_path))
%Generate and save plots to output_path
%First generate individual plots from each row of the pck_curve_array
colormap default;

for i = 1:length(error_data)
    all_plot = [];
    for j = 1:ng+1
        figure(1);
        cla;
        plot(thresh,pck_curve_array{i,j},'LineWidth',2);
        all_plot = [all_plot; pck_curve_array{i,j}];
        axis([0 150 0 1]);
        title([pck_table{1,j+1} '  PCK150mm']);
        output_dir = [output_base_path filesep error_data(i).method];
        if(exist(output_dir,'dir') ~= 7)
            mkdir(output_dir);
        end
        saveas(gcf,[output_dir filesep pck_table{1,j+1}], 'fig');
        saveas(gcf,[output_dir filesep pck_table{1,j+1}], 'svg');
        saveas(gcf,[output_dir filesep pck_table{1,j+1}], 'png');
        
    end
    figure(2);
    cla;
    plot(thresh,all_plot,'LineWidth',2);
    axis([0 150 0 1]);
    hold off;
    legend(pck_table(1,2:end));
    saveas(gcf,[output_dir filesep 'All'], 'fig');
    saveas(gcf,[output_dir filesep 'All'], 'svg');
    saveas(gcf,[output_dir filesep 'All'], 'png');
end
end

end
%Then group the plots by methods 

