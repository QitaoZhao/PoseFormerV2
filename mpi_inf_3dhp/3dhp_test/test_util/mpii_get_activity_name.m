function [activity_names] = mpii_get_activity_name(activity_id)

activities{1} = 'Standing/Walking';
activities{2} = 'Exercising';
activities{3} = 'Sitting';
activities{4} = 'Reaching/Crouching';
activities{5} = 'On The Floor';
activities{6} = 'Sports';
activities{7} = 'Miscellaneous';

activity_names = activities(activity_id);
end