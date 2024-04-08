import numpy
import numpy as np


def i2t_gl(full_img_emb_aggrs, full_cap_emb_aggrs, img_embs, cap_embs, img_emb_fusions, cab_emb_fusions, img_lenghts,
           cap_lenghts, npts=None, return_ranks=True, ndcg_scorer=None, fold_index=0, measure='dot', sim_function=None,
           sim_function_new=None, cap_batches=1, pl_module=None, fold5=-1, topk=None, weight=None):
    # global_sims = np.matmul(full_img_emb_aggrs, full_cap_emb_aggrs.T)  #

    if npts is None:
        npts = img_embs.shape[0] // 5

    index_list = []
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    full_img_emb_aggrs = np.array([full_img_emb_aggrs[i] for i in range(0, len(full_img_emb_aggrs), 5)])
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    img_emb_fusions = np.array([img_emb_fusions[i] for i in range(0, len(img_emb_fusions), 5)])

    global_sims = np.matmul(full_img_emb_aggrs, full_cap_emb_aggrs.T)  # (5N, 5N)
    sims = sim_function_new(img_embs, cap_embs, cap_lenghts, pl_module)
    final_sims = global_sims * weight + sims * (1 - weight)  # (N, 5N)

    for index in range(npts):

        d_r = final_sims[index]
        g_inds = numpy.argsort(d_r)[::-1]
        top_g_inds = list(g_inds[0:topk])

        im = img_embs[index].reshape(1, img_embs.shape[1], img_embs.shape[2])  # (1, N, dim)
        im_f = img_emb_fusions[index].reshape(1, img_emb_fusions.shape[1], img_emb_fusions.shape[2])  # (1, N, dim)

        cap_now = cap_embs[top_g_inds]
        cap_f_now = cab_emb_fusions[top_g_inds]
        cap_lenghts_now = list(cap_lenghts[top_g_inds])  # (150, )

        # d: (1, 150)
        # d_l_1 = final_sims[5 * index][top_g_inds]
        d_l_1 = d_r[top_g_inds]
        d_l_2 = sim_function(im_f, cap_now, cap_lenghts_now, 9)
        d_l_3 = sim_function(im, cap_f_now, cap_lenghts_now, 9)
        d_f = (d_l_1 + d_l_2 + d_l_3).flatten()

        l_inds = numpy.argsort(d_f)[::-1]
        inds = g_inds[l_inds]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        tmp_inss = []
        for i in range(5 * index, 5 * index + 5, 1):
            tmp_ins = list(numpy.where(inds == i))
            if len(tmp_ins[0]) <= 0:
                continue
            tmp_inss.append(tmp_ins[0][0])
        if len(tmp_inss) <= 0:
            tmp_inss.append(0)
        tmp = min(tmp_inss)
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, 0., 0.), (ranks, top1), sims
    else:
        return (r1, r5, r10, medr, meanr, 0., 0.), sims


def t2i_gl(full_img_emb_aggrs, full_cap_emb_aggrs, img_embs, cap_embs, img_emb_fusions, cab_emb_fusions, img_lenghts,
           cap_lenghts, npts=None, return_ranks=True, ndcg_scorer=None, fold_index=0, measure='dot', sim_function=None,
           sim_function_new=None, cap_batches=1, pl_module=None, sims=None, topk=None, weight=None):
    # global_sims = np.matmul(full_img_emb_aggrs, full_cap_emb_aggrs.T)  #

    if npts is None:
        npts = img_embs.shape[0] // 5

    index_list = []
    ims = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    f_ims = np.array([img_emb_fusions[i] for i in range(0, len(img_emb_fusions), 5)])
    full_img_emb_aggrs = np.array([full_img_emb_aggrs[i] for i in range(0, len(full_img_emb_aggrs), 5)])

    ranks = numpy.zeros(5 * npts)
    top50 = numpy.zeros((5 * npts, 5))

    global_sims = np.matmul(full_img_emb_aggrs, full_cap_emb_aggrs.T).T  # (N, 5N) -> (5N, N)
    # sims = sim_function_new(ims, cap_embs, cap_lenghts, pl_module).T
    # sims = (np.array([sims[i] for i in range(0, len(sims), 5)])).T
    sims = (sims).T

    final_sims = sims * (1 - weight) + global_sims * weight

    for index in range(npts):

        # Get query captions
        queries_1 = cap_embs[5 * index:5 * index + 5]
        queries_2 = cab_emb_fusions[5 * index:5 * index + 5]
        queries_len = cap_lenghts[5 * index:5 * index + 5]

        d_r = final_sims[5 * index: 5 * index + 5]
        inds = numpy.zeros((len(d_r), topk))  # (5, 150)

        for i in range(len(d_r)):
            di_g = d_r[i]
            g_inds = numpy.argsort(di_g)[::-1]
            cap_inds = list(g_inds[:topk])

            quer = queries_1[i].reshape(1, queries_1[i].shape[0], queries_1[i].shape[1])  # (1, N, dim)
            f_quer = queries_2[i].reshape(1, queries_2[i].shape[0], queries_2[i].shape[1])  # (1, N, dim)

            quer_len = [queries_len[i]]

            ims_now = ims[cap_inds]
            f_ims_now = f_ims[cap_inds]

            # (1, 150)
            d_l_1 = d_r[i][cap_inds]
            # d_l_1 = sim_function_new(ims_now, quer, quer_len, pl_module).T
            d_l_2 = sim_function(f_ims_now, quer, quer_len, 9).T
            d_l_3 = sim_function(ims_now, f_quer, quer_len, 9).T
            d_f = (d_l_1 + d_l_2 + d_l_3).flatten()

            l_inds = numpy.argsort(d_f)[::-1]
            inds[i] = g_inds[l_inds]
            r_r = numpy.where(inds[i] == index)[0]
            if len(r_r) <= 0:
                ranks[5 * index + i] = 0
            else:
                ranks[5 * index + i] = r_r[0]
            top50[5 * index + i] = inds[i][0:5]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr, 0., 0.), (ranks, top50)
    else:
        return (r1, r5, r10, medr, meanr, 0., 0.)
