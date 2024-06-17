test_subject_id = [1,2,3,4,5,6];
focalL{1} = (2048/10)*7.320339203; % res/sensorsize*focalLengthMM
focalL{2} = focalL{1};
focalL{3} = focalL{1};
focalL{4} = focalL{1};
focalL{5} = (1920/10)*8.770747185; % res/sensorsize*focalLengthMM
focalL{6} = focalL{5};

for ts = 1:6

           %Fancy predictions here: predict_2d and predict_3d. predict_2d is in the uncropped(?) image space

            focalLengthInPX = focalL{ts};

            resolutionXInPX = image_size{ts}(2);  %I can't seem to remember which one is x or why. Try both until something works :)
            resolutionYInPX = image_size{ts}(1);
            principalPointX = resolutionXInPX/2;
            principalPointY = resolutionYInPX/2;
            center = predict_2d(15,:) - [principalPointX, principalPointY]; % (pelvis location)
            R = mpii_perspective_correction(center(1), 0, focalLengthInPX);
            predict_3d = R * predict_3d;
end