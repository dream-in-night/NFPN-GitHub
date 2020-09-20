# _base_ = '../fast_rcnn/fast_rcnn_r50_fpn_1x_coco.py'
_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=dict(
            type='BFP4',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local'
            ),
    roi_head=dict(
    	bbox_head=dict(
    		loss_bbox=dict(
    			_delete_=True,
	            type='BalancedL1Loss',
	            alpha=0.5,
	            gamma=1.5,
	            beta=1.0,
	            loss_weight=1.0)
    		)
    	)
    )
