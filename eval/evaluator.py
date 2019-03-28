import torch
from utils import to_torch
from .eva_functions import evaluate
import numpy as np
from torch import nn


def evaluate_seq(distmat, query_pids, query_camids, gallery_pids, gallery_camids, cmc_topk=[1, 5, 10, 20]):
    query_ids = np.array(query_pids)
    gallery_ids = np.array(gallery_pids)
    query_cams = np.array(query_camids)
    gallery_cams = np.array(gallery_camids)

    cmc_scores, mAP = evaluate(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    for r in cmc_topk:
        print("Rank-{:<3}: {:.1%}".format(r, cmc_scores[r-1]))
    print("------------------")

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores[0]


def pairwise_distance_tensor(query_x, gallery_x):

    m, n = query_x.size(0), gallery_x.size(0)
    x = query_x.view(m, -1)
    y = gallery_x.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) +\
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    return dist


class Evaluator(object):

    def __init__(self, cnn_model):
        super(Evaluator, self).__init__()
        self.cnn_model = cnn_model
        self.softmax = nn.Softmax(dim=-1)

    def extract_feature(self, data_loader):  # 2
        # print_freq = 50
        self.cnn_model.eval()

        qf = []
        # qf_raw = []

        for i, inputs in enumerate(data_loader):
            imgs, _, _ = inputs
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b*n, s, c, h, w)
            imgs = to_torch(imgs)  # torch.Size([8, 8, 3, 256, 128])
            # flows = to_torch(flows)  # torch.Size([8, 8, 3, 256, 128])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            imgs = imgs.to(device)
            # flows = flows.to(device)
            with torch.no_grad():
                out_feat, out_raw = self.cnn_model(imgs)
                allfeatures = out_feat.view(n, -1)  # torch.Size([8, 128])
                # allfeatures_raw = out_raw.view(n, -1)  # torch.Size([8, 128])
                allfeatures = torch.mean(allfeatures, 0).data.cpu()  # 汇总一个序列特征,取平均
                # allfeatures_raw = torch.mean(allfeatures_raw, 0).data
                qf.append(allfeatures)
                # qf_raw.append(allfeatures_raw)
        qf = torch.stack(qf)
        #    qf_raw = torch.stack(allfeatures_raw)

        print("Extracted features for query/gallery set, obtained {}-by-{} matrix"
              .format(qf.size(0), qf.size(1)))
        return qf

    def evaluate(self, query_loader, gallery_loader, queryinfo, galleryinfo):
        # 1
        self.cnn_model.eval()

        querypid = queryinfo.pid  # 100 ge id  <class 'list'>: [74, 20, 90, 151, 1, 69, 84, 149, 5, 111, -1, 154, ...]
        querycamid = queryinfo.camid  # 00000000000

        gallerypid = galleryinfo.pid   # <class 'list'>: [74, 20, 90, 151, 1, 69, 84, 149, 5, 111, -1, 154, ...]
        gallerycamid = galleryinfo.camid  # 1111111111

        pooled_probe = self.extract_feature(query_loader)  # 1980 * 128
        pooled_gallery = self.extract_feature(gallery_loader)
        print("Computing distance matrix")
        distmat = pairwise_distance_tensor(pooled_probe, pooled_gallery)

        return evaluate_seq(distmat, querypid, querycamid, gallerypid, gallerycamid)
