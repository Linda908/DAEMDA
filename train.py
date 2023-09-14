import datetime

from model import *
from preData import *

def fold_valid(args):
    similarity_feature = similarity_feature_process(args)
    edge_idx_dict, g = load_fold_data(args)
    n_rna = edge_idx_dict['true_md'].shape[0]
    n_dis = edge_idx_dict['true_md'].shape[1]

    metric_result_list = []
    metric_result_list_str = []
    metric_result_list_str.append('AUC    AUPR    Acc    F1    pre    recall')
    for i in range(args.kfolds):
        model = G_Module(args, n_rna, n_dis).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

        print(f'###########################Fold {i + 1} of {args.kfolds}###########################')
        Record_res = []
        fold_y = []
        Record_res.append('AUC    AUPR    Acc    F1    pre    recall')
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()
            out = model(args, similarity_feature, edge_idx_dict[str(i)]['fold_train_edges_80p_80n'],
                        edge_idx_dict[str(i)]['fold_train_edges_80p_80n'], g[str(i)]["fold_train_edges_80p_80n"]).view(
                -1)
            loss = criterion(out, edge_idx_dict[str(i)]['fold_train_label_80p_80n'])
            loss.backward()
            optimizer.step()
            # validation
            test_auc, metric_result, y_true, y_score = valid_fold(model,
                                                                  similarity_feature,
                                                                  edge_idx_dict[str(i)]['fold_train_edges_80p_80n'],
                                                                  edge_idx_dict[str(i)]['fold_valid_edges_20p_20n'],
                                                                  g[str(i)]["fold_train_edges_80p_80n"],
                                                                  edge_idx_dict[str(i)]['fold_valid_label_20p_20n'],
                                                                  args)
            # 记录单次实验结果
            One_epoch_metric = '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*metric_result)
            Record_res.append(One_epoch_metric)
            # 当100次结束后保留实验结果以及实验原始数据
            if epoch + 1 == args.epoch:
                metric_result_list.append(metric_result)
                metric_result_list_str.append(One_epoch_metric)
                fold_y = [str(y_true.cpu().numpy().tolist()), str(y_score.cpu().numpy().tolist())]
            # 打印单次实验的loss和验证集auc
            print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} '.format(epoch, loss.item(), test_auc))

    arr = np.array(metric_result_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    metric_result_list_str.append('平均值：')
    metric_result_list_str.append('{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*list(averages)))
    metric_result_list_str.append(str(args.GATh))
    metric_result_list_str.append(str(args.head))

    now = datetime.datetime.now()
    with open('平均_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '_.txt', 'w') as f:
        f.write('\n'.join(metric_result_list_str))
    return averages


def valid_fold(model, data, encodeEdge, decodeEdge, g, lable, args):
    model.eval()
    with torch.no_grad():
        z = model.encode(args, data, encodeEdge, g)
        out = model.decode(z, decodeEdge).view(-1).sigmoid()
        model.train()
    metric_result = caculate_metrics(lable.cpu().numpy(), out.cpu().numpy())
    my_acu = metrics.roc_auc_score(lable.cpu().numpy(), out.cpu().numpy())
    return my_acu, metric_result, lable, out
