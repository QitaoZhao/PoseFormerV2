function [joint_idx, joint_parents_o1, joint_parents_o2, joint_names] = mpii_get_joints(joint_set_name)

    original_joint_idx = [10, 13, 16, 19, 22, 25, 28, 29, 31, 36, 40, 42, 43, 45, 50, 54, 56, 57,  63, 64, 69, 70, 71, 77, 78, 83, 84, 85];              %                            
        
    original_joint_names = {'spine3', 'spine4', 'spine2', 'spine1', 'spine', ...     %5       
                        'neck', 'head', 'head_top', 'left_shoulder', 'left_arm', 'left_forearm', ... %11
                       'left_hand', 'left_hand_ee',  'right_shoulder', 'right_arm', 'right_forearm', 'right_hand', ... %17
                       'right_hand_ee', 'left_leg_up', 'left_leg', 'left_foot', 'left_toe', 'left_ee', ...        %23   
                       'right_leg_up' , 'right_leg', 'right_foot', 'right_toe', 'right_ee'};  
                   
                   
    all_joint_names = {'spine3', 'spine4', 'spine2', 'spine', 'pelvis', ...     %5       
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', ... %11
       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17
       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23   
       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'}; 
   
   
  %The O1 and O2 indices are relaive to the joint_idx, regardless of the joint set 
                   
switch joint_set_name
    %For internal use only!!!
    case 'original'  %%These give the original indices from the dumped out mddd file, the remaining joint sets are wrt the 'all' labels
        joint_idx = original_joint_idx;              %                            
        joint_parents_o1 = [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19, 20, 21, 22, 5, 24, 25, 26, 27 ];
        joint_parents_o2 = [4, 3, 5, 5, 5, 1, 2, 6, 2, 6, 9, 10, 11, 2, 6, 14, 15, 16, 4, 5, 19, 20, 21, 4, 5, 24, 25, 26];
        joint_names = original_joint_names;  
    %Use joint sets from here    
    case 'all'
        joint_idx = 1:28;              %These index into the joints extracted in the original set
        joint_parents_o1 = [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19, 20, 21, 22, 5, 24, 25, 26, 27 ];
        joint_parents_o2 = [4, 3, 5, 5, 5, 1, 2, 6, 2, 6, 9, 10, 11, 2, 6, 14, 15, 16, 4, 5, 19, 20, 21, 4, 5, 24, 25, 26];
        joint_names = all_joint_names;
        
    case 'cpm'  %CPM Joints in CPM Order
        joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5];
        joint_parents_o1 = [ 2, 15, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15];
        joint_parents_o2 = [15, 15, 15, 2, 3, 15, 2, 6, 2, 15, 9, 2, 15, 12, 15];  
        joint_names = all_joint_names(joint_idx);
        
    case 'relevant' %Human3.6m joints in CPM order
        joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7];
        joint_parents_o1 = [ 2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2];
        joint_parents_o2 = [ 16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16];
        joint_names = all_joint_names(joint_idx);
             
    otherwise
end
end
