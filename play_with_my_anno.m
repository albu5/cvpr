annodir = 'my_anno';
annofmt = 'data_%2.2d.mat';

mkdir group_int
grpmatfmt = 'group_int/group_int%2.2d.mat';
maskmatfmt = 'group_int/mask_int%2.2d.mat';

max_people = 20;

for i = 1:33
    annostr = fullfile(annodir, sprintf(annofmt, i));
    anno = load(annostr);
    anno = anno.anno_data;
    n_people = numel(anno.people);
    group_int = zeros([n_people, n_people, anno.nframe]);
    mask_int = ones([max_people, max_people, anno.nframe]);
    
    for t = 1:anno.nframe
        % form group belonging matrix
        group_label = anno.groups.grp_label(t,:);
        group_label = repmat(group_label, [n_people, 1]);
        group_int(:,:,t) = group_label == group_label';
        
        % form a mask indicating which persons ae not in the scene
        for person = 1:max_people
            if person > n_people
                mask_int(person,:,t) = 0;
                mask_int(:,person,t) = 0;
            elseif anno.groups.grp_label(t,person) == 0
                mask_int(person,:,t) = 0;
                mask_int(:,person,t) = 0;
            end
        end
    end
    save(sprintf(grpmatfmt, i), 'group_int');
    save(sprintf(maskmatfmt, i), 'mask_int');
    
    display(sprintf(grpmatfmt, i))
end
