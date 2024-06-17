function [sequencewise_table, activitywise_table] = mpii_evaluate_errors(sequencewise_error, sequencewise_activity)

joint_groups = mpii_get_pck_auc_joint_groups();
[~,~,~,joint_names] = mpii_get_joints('relevant');
all_errors = [];
all_activities = [];
sequencewise_pck = {};
sequencewise_auc = {};
nj = length(joint_names);
sequencewise_mpjpe = cell(length(sequencewise_error)+1,nj+2);
sequencewise_mpjpe(1,2:(nj+1)) = joint_names;
sequencewise_mpjpe{1,(nj+2)} = 'Average';
 %Generate MPJPE and PCK/AUC By sequence first
 %error_dat = {};
 %delete('error_dat');
 for i = 1:length(sequencewise_error)
     if(isempty(all_errors))
         all_errors = sequencewise_error{i}(:,1,:);
     else
         all_errors = cat(3,all_errors, sequencewise_error{i}(:,1,:));
     end
     all_activities = [all_activities; sequencewise_activity{i}(:)];
         
     error_dat(i) = mpii_3D_error(['TestSeq' int2str(i)], sequencewise_error{i}(:,1,:));
     sequencewise_mpjpe{i+1,1}= ['TestSeq' int2str(i)];
     mpjpe = mean(sequencewise_error{i}(:,1,:),3);
     sequencewise_mpjpe(i+1,2:(nj+1)) = num2cell(mpjpe');
     sequencewise_mpjpe{i+1,(nj+2)} = mean(mpjpe(:));
 end
 [pck, auc] = mpii_compute_3d_pck(error_dat, joint_groups, []);
 sequencewise_pck = [sequencewise_pck; pck];
 sequencewise_pck{1,1} = 'PCK';
 sequencewise_auc = [sequencewise_auc; auc];
 sequencewise_auc{1,1} = 'AUC';

 activitywise_pck = {};
activitywise_auc = {};
activitywise_mpjpe = cell(7+2,nj+2);
activitywise_mpjpe(1,2:(nj+1)) = joint_names;
activitywise_mpjpe{1,(nj+2)} = 'Average';
 %Generate MPJPE and PCK/AUC By activity
 %error_dat = {};
 clear('error_dat');
 for i = 1:7
     error_dat(i) = mpii_3D_error(mpii_get_activity_name(i), all_errors(:,:,all_activities == i));
     activitywise_mpjpe{i+1,1} = mpii_get_activity_name(i);
     mpjpe = mean(all_errors(:,:,all_activities == i),3);
     activitywise_mpjpe(i+1,2:(nj+1)) = num2cell(mpjpe');
     activitywise_mpjpe{i+1,(nj+2)} = mean(mpjpe(:));
 end
 overall_mpjpe  = mean(all_errors,3);
 activitywise_mpjpe{end,1} = 'All';
 activitywise_mpjpe(end,2:(nj+1)) = num2cell(overall_mpjpe');
 activitywise_mpjpe{end,(nj+2)} = mean(overall_mpjpe(:));
[pck, auc] = mpii_compute_3d_pck(error_dat, joint_groups, []);
 activitywise_pck = [activitywise_pck; pck];
 activitywise_pck{1,1} = 'PCK';
 activitywise_auc = [activitywise_auc; auc];
 activitywise_auc{1,1} = 'AUC';
 clear('error_dat');
 error_dat(1) = mpii_3D_error('All', all_errors);
[pck, auc] = mpii_compute_3d_pck(error_dat, joint_groups, []);
activitywise_pck = [activitywise_pck; pck(2:end,:)];
activitywise_auc = [activitywise_auc; auc(2:end,:)];
     
sequencewise_table = sequencewise_mpjpe;
sequencewise_table(size(sequencewise_table,1)+1:size(sequencewise_table,1)+size(sequencewise_pck,1),1:size(sequencewise_pck,2)) = sequencewise_pck;
sequencewise_table(size(sequencewise_table,1)+1:size(sequencewise_table,1)+size(sequencewise_auc,1),1:size(sequencewise_auc,2)) = sequencewise_auc;
activitywise_table = activitywise_mpjpe;
activitywise_table(size(activitywise_table,1)+1:size(activitywise_table,1)+size(activitywise_pck,1),1:size(activitywise_pck,2)) = activitywise_pck;
activitywise_table(size(activitywise_table,1)+1:size(activitywise_table,1)+size(activitywise_auc,1),1:size(activitywise_auc,2)) = activitywise_auc;


     
end