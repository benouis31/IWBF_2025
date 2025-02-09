import torch
from sklearn import preprocessing, metrics
from itertools import combinations
import numpy as np
import sklearn
import datetime
from utils_plot import FullPairComparer,AverageMeter, evaluate, plot_roc, plot_DET_with_EER, plot_density,accuracy

outdir='./IWBF_2025/results'



def test(model, dataloader,epoch,output,SEED,ois_graph=False):
    global best_test
    labels, distances = [], []
    with torch.set_grad_enabled(False):
        comparer = FullPairComparer()
        model.eval()
        #for batch_idx, (data1, data2, target) in enumerate(dataloader):
        fpr_list, fnr_list, fpr_optimum_list, fnr_optimum_list = [], [], [], []

        for batch_idx,(data1,data2, target) in enumerate(dataloader):
            dist = []
            #target = target.cuda(non_blocking=True)
            #data1,data2=data
            target = target
            data1=data1.permute(0, 2, 1)
            data2=data2.permute(0, 2, 1)


            output1 = model(data1,False)
            output2 = model(data2,False)
            print(output1.shape)
            #dist=F.pairwise_distance(output1, output2)
            dist = comparer(output1, output2) #TODO: sign - torch.sign()
            #dist = comparer(torch.sign(F.relu(output1)), torch.sign(F.relu(output2)))  # TODO: sign - torch.sign()
            distances.append(dist.data.cpu().numpy())
            labels.append(target.data.cpu().numpy())
            if batch_idx % 50 == 0:
                print('Batch-Index -{}'.format(str(batch_idx)))


    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    #distances = normalize_scores(distances)
    tpr, fpr, fnr, fpr_optimum, fnr_optimum, accuracy, threshold = evaluate(distances, labels)
    #fpr, tpr, thresholds = metrics.roc_curve(labels, distances, pos_label=1)
    #roc_metrics, metrics_threds, AUC = compute_roc_metrics(fpr, tpr, thresholds)

    EER = np.mean(fpr_optimum + fnr_optimum) / 2
    th,eer=optimize_threshold_for_eer(distances, labels, num_thresholds=100)
    from sklearn.metrics import roc_curve, auc
    #fpr, tpr, thresholds = roc_curve(distances, labels)
   # roc_auc = auc(fpr, tpr)
    print('tpr',tpr)
    print('fpr',fpr)
    print('fpr_optimum',fpr_optimum)
    print('fnr_optimum',fnr_optimum)
    #EEER=calculate_eer(fpr, fnr)
    print('TEST - Accuracy           = {:.12f}'.format(accuracy))
    print('TEST - EER                = {:.12f}'.format(EER))
    print('TEST - eer                = {:.12f}'.format(eer))
    is_best = EER <= best_test
    best_test = min(EER, best_test)
    #save_metrics_to_csv(fpr, fnr, fpr_optimum, fnr_optimum, epoch)

    if is_best and is_graph:
        plot_roc(fpr, tpr, figure_name=output + '/Test_ROC-{}.png'.format(SEED))
        plot_DET_with_EER(fpr, fnr, fpr_optimum, fnr_optimum,
                          figure_name=output + '/Test_DET-{}.png'.format(SEED))
        plot_density(distances, labels, figure_name=output + '/Test_DENSITY-{}.png'.format(SEED))
       


        if args.evaluate is False:
            shutil.copyfile(output + '/model_best.pth.tar', output + '/test_model_best.pth.tar')

    return EER



if __name__ == '__main__':
    device = 'cpu:1'
    data_type = 'WESAD'
    data_path = f"./data/{data_type}"
    path_1= '/IWBF_2025/data/VerBIO-norm.hdf5'
    path_2= '/IWBF_2025/data/WESAD-norm.hdf5'
    output=f".IWBF_2025/results/{SEED}/"

    train_mode = 'fine-tuned' #Frozen
    
    config_module = __import__(f'config_files.{data_type}_Configs', fromlist=['Config'])
    configs = config_module.Config()
    
    F1 = []
    ACC = []
    for SEED in SEEDS:
        #train_dl, valid_dl, test_dl = data_generator(data_path, configs, 'supervised')
        train_dl, valid_dl, test_dl = data_generator(path_1,path_2, configs, train_mode)
        
        
        

        ckpt = f'./IWBF_2025/checkpoints/{data_type}/{train_mode}_{SEED}.pth'
        model = TMS_BIO(configs)
  
       
        model.load_state_dict(torch.load(ckpt))
        #new_state_dict = {k: v for k, v in state_dict.items() if k not in ['linear.weight', 'linear.bias']}
        #model.load_state_dict(new_state_dict, strict=False)
        
        model = model.to(device)
        model.eval()
        EER = test(model, test_dl, output,SEED,is_graph=True)
        


