Details:
The test set has 6 sequences and a dedicated folder for each sequence.
Each folder contains a .mat file with the following information

valid_frame: Indicates whether the frame is valid or not. Invalid frames
won't be used for evaluation. Refer to mpii_test_predictions.m for more.

activity_annnotation: Activity annotations per frame, used for
generating activitywise error reports

univ_annot3: A 3x17x1xn matrix containing the 3D annotations in mm for
17 joints. The annotations are scaled to the height of the universal
skeleton used by Human3.6m. The order and the names of the joints can be
observed using [~,~,~,joint_names] = mpii_get_joints('relevant');

The file mpii_test_predictions.m should be a helpful starting point.
Additionally, information about the crops (the original size in the frame)
is available in the mat/zip file attached with the email. 

If you want to evaluate by scene setting, you can use the sequencewise evaluation
to convert to these numbers by doing  
#1:Studio with Green Screen (TS1*603 + TS2 *540)/ (603+540)    
#2:Studio without Green Screen (TS3*505+TS4*553)/(505+553)   
#3:Outdoor (TS5*276+TS6*452)/(276+452)
