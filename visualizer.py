import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from vis_utils import ArgoMapVisualizer


class VisualizerDsp():
    def __init__(self):
        self.map_vis = ArgoMapVisualizer()

    def draw_once(self, post_out, data, show_map=False, test_mode=False):

        batch_size = len(data['argo_id'])

        argo_id = data['argo_id'][0]
        city = data['city'][0]
        orig = data['orig'][0].cpu().detach().numpy()
        rot = data['rot'][0].cpu().detach().numpy()

        trajs_obs = data['used_trajs'][0].cpu().detach().numpy()
        trajs_fut = data['gt_preds'][0].cpu().detach().numpy()
        # graph_da = data['MLG_DA'][0]
        # graph_ls = data['MLG_LS'][0]
        # pairs_da2ls = data['MLG_DA2LS'][0]
        # goda_idcs = data['GODA_IDCS'][0]
        # goda_label = data['GODA_LABEL'][0]

        # goda_cls = post_out['goda_cls'][:graph_da['num_nodes']].cpu().detach().numpy()
        traj_pred = post_out['reg'][0].cpu().detach().numpy()

        #这里主要看有没有refine
        mid_pred = post_out['test_ctrs'][0][0][0,:,:].cpu().detach().numpy()
        # mid_pred_refine = post_out['test_ctrs_refined'][0][0].cpu().detach().numpy()
        

        # goal_pred = post_out['goal_pred'][0].cpu().detach().numpy()
        prob_pred = post_out['cls'][0].cpu().detach().numpy()
        prob_mid = post_out['test_cls'][0].cpu().detach().numpy()

        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        plt.axis('off')
        # ax.set_title('argo_id: {}-{}'.format(argo_id, city))

        if show_map:
            self.map_vis.show_surrounding_elements(ax, city, orig)
        else:
            rot = np.eye(2)
            orig = np.zeros(2)


        for i, traj in enumerate(trajs_obs):
            zorder = 10
            if i == 0:
                clr = 'grey'
                zorder = 20
            # elif i == 1:
            #     clr = 'gold'
            else:
                # clr = 'grey'
                break
            ax.plot(traj[:, 0], traj[:, 1], marker='.', alpha=0.5, color=clr, zorder=zorder)
            # ax.scatter(traj[:, 0], traj[:, 1], s=list(traj[:, 2] * 50 + 1), color='b')

        if not test_mode:
            # trajs_fut = trajs_fut.dot(rot.T) + orig
            for i, traj in enumerate(trajs_fut):
                zorder = 10
                if i == 0:
                    clr = 'r'
                    zorder = 20
                # elif i == 1:
                #     clr = 'gold'
                else:
                    break
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, color=clr, linewidth=3, marker='.', zorder=zorder)
                ax.plot(traj[-1, 0], traj[-1, 1], alpha=0.5, color=clr, marker='o', zorder=zorder, markersize=10)

        for i, trajs in enumerate(traj_pred):
            if i==0:
                for j, traj in enumerate(trajs):
                    ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, color='g', linewidth=3, marker='.', zorder=20)
                    ax.plot(traj[-1, 0], traj[-1, 1], marker='*', color='g', markersize=10, alpha=0.75, zorder=20)
            else:
                break
                # ax.text(traj[-1, 0], traj[-1, 1], '{:.2f}'.format(prob_pred[i][j]), zorder=15)


        # #方法二，自适应的,中间点
        # traj = trajs_fut[0]
        # last = traj[0,:]
        # for i in range(mid_pred.shape[0]):
        #     distance = np.linalg.norm(last - mid_pred[i])
        #     if distance < 3:
        #         distance = 3
        #     circle = plt.Circle((mid_pred[i, 0], mid_pred[i, 1]), distance, color="orange", zorder=40, alpha=0.5)
        #     ax.add_artist(circle)
        #     ax.plot(mid_pred[i, 0], mid_pred[i, 1], color='orange', marker='.', zorder=50, markersize=10)
        #     last = mid_pred[i]


        #方法三，自适应的，中间点和真实轨迹求中间值
        traj = trajs_fut[0]
        traj_1 = traj[9::10]
        mid_pred = (mid_pred + traj_1)/2
        for i in range(mid_pred.shape[0]):
            ax.plot(mid_pred[i, 0], mid_pred[i, 1], alpha=0.5, color='orange', marker='o', zorder=40, markersize=40)
            ax.plot(mid_pred[i, 0], mid_pred[i, 1], color='orange', marker='.', zorder=50, markersize=10)

        plt.savefig('./vis/LaneGCN/'+str(argo_id)+'.svg')
        plt.close()  # 关闭所有打开的图形

        # plt.show()

def restore_traj(feats, orig, rot):
    restored_trajs = []
    inv_rot = rot.T  # 旋转矩阵的逆矩阵是它的转置

    for feat in feats:
        # Step 1: 累积差分还原
        traj_cumsum = np.zeros_like(feat[:, :2])  # 初始化原始轨迹
        traj_cumsum[0] = feat[0, :2]
        for i in range(1, len(feat)):
            traj_cumsum[i] = traj_cumsum[i - 1] + feat[i, :2]

        # Step 2: 逆旋转还原
        traj_rot_reversed = np.matmul(inv_rot, traj_cumsum.T).T

        # Step 3: 加回平移
        traj_original = traj_rot_reversed + orig.reshape(-1, 2)

        # 保存还原的轨迹
        restored_trajs.append(traj_original)

    return np.array(restored_trajs)
